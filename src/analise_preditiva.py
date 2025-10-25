"""
Módulo para análise preditiva e modelagem de dados logísticos.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class AnalisePreditiva:
    """Classe responsável pela análise preditiva e modelagem."""
    
    def __init__(self, dados_logistica, output_path='output'):
        """
        Inicializa a análise preditiva.
        
        Args:
            dados_logistica (pd.DataFrame): Dados de logística diários
            output_path (str): Caminho para salvar os resultados
        """
        self.dados_logistica = dados_logistica.copy()
        self.output_path = output_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.resultados = {}
        self.melhor_modelo_nome = None
        self.melhor_modelo = None
        
        # Criar pastas de output
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'modelos'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'graficos'), exist_ok=True)
        
        # Configurar estilo dos gráficos
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (14, 6)
    
    def engenharia_features(self, dados_mensais=None, use_margem_as_feature=False):
        """
        Realiza engenharia de features para o modelo.
        
        Args:
            dados_mensais (pd.DataFrame): Dados mensais para merge
            use_margem_as_feature (bool): Se deve incluir margem como feature
        """
        print("🔧 Realizando engenharia de features...")
        
        df = self.dados_logistica.copy().sort_values('Data').reset_index(drop=True)
        
        # Temporais simples
        df['DiaSemana'] = df['Data'].dt.dayofweek
        df['DiaMes'] = df['Data'].dt.day
        df['MesNum'] = df['Data'].dt.month
        df['Trimestre'] = df['Data'].dt.quarter
        df['IsWeekend'] = df['DiaSemana'].isin([5,6]).astype(int)
        df['IsMonthStart'] = df['Data'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Data'].dt.is_month_end.astype(int)
        
        # Ratios e eficiências (diárias)
        df['Eficiencia_Entrega'] = df['Entregas'] / df['KM Percorridos']
        df['Peso_por_Entrega'] = df['Peso (ton)'] / df['Entregas']
        df['Frete_por_KM'] = df['Frete'] / df['KM Percorridos']
        df['Frete_por_Entrega'] = df['Frete'] / df['Entregas']
        df['Custo_por_Entrega'] = df['Custo Total'] / df['Entregas']
        df['Custo_por_Ton'] = df['Custo Total'] / df['Peso (ton)']
        df['Custo_por_KM'] = df['Custo/KM']
        df['Prop_Combustivel'] = df['Custo Combustível'] / df['Custo Total']
        df['Prop_Manutencao'] = df['Custo Manutenção'] / df['Custo Total']
        df['Prop_Motorista'] = df['Custo Motorista'] / df['Custo Total']
        df['Peso_por_KM'] = df['Peso (ton)'] / df['KM Percorridos']
        
        # Agregar informações mensais se disponível
        if dados_mensais is not None:
            mensal = dados_mensais.copy()
            mensal = mensal.rename(columns={
                'Custo Total': 'M_Custo_Total', 'Custo Combustível': 'M_Custo_Combustivel',
                'Custo Manutenção':'M_Custo_Manutencao','Custo Motorista':'M_Custo_Motorista',
                'Frete':'M_Frete','Margem':'M_Margem','KM Percorridos':'M_KM_Percorridos',
                'Entregas':'M_Entregas','Peso (ton)':'M_Peso_ton','Margem %':'M_Margem_pct',
                'Custo/KM':'M_Custo_KM'
            })
            df = df.merge(mensal[['Mês'] + [c for c in mensal.columns if c != 'Mês']], 
                         on='Mês', how='left')
        
        # Rolling features
        df = df.sort_values('Data').reset_index(drop=True)
        for wnd in [7, 30]:
            df[f'roll_mean_CustoTotal_{wnd}d'] = df['Custo Total'].rolling(window=wnd, min_periods=1).mean()
            df[f'roll_mean_KM_{wnd}d'] = df['KM Percorridos'].rolling(window=wnd, min_periods=1).mean()
            df[f'roll_mean_Entregas_{wnd}d'] = df['Entregas'].rolling(window=wnd, min_periods=1).mean()
            df[f'roll_mean_CustoKM_{wnd}d'] = df['Custo/KM'].rolling(window=wnd, min_periods=1).mean()
        
        # Lag features
        for lag in [1, 7]:
            df[f'lag_Entregas_{lag}d'] = df['Entregas'].shift(lag)
            df[f'lag_CustoTotal_{lag}d'] = df['Custo Total'].shift(lag)
            df[f'lag_CustoKM_{lag}d'] = df['Custo/KM'].shift(lag)
        
        # Preencher NA
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        # Remover margem se não deve ser usada como feature
        if not use_margem_as_feature and 'Margem' in df.columns:
            df.drop(columns=['Margem'], inplace=True)
        
        self.df = df
        
        # Salvar versão com features
        df.to_csv(os.path.join(self.output_path, 'dados_logistica_features.csv'), index=False)
        print(f"✅ Features criadas. Total de registros: {len(df)}")
        print(f"📁 Arquivo com features salvo em {os.path.join(self.output_path, 'dados_logistica_features.csv')}")
    
    def preparar_dados(self, target='Margem %'):
        """
        Prepara os dados para treinamento.
        
        Args:
            target (str): Coluna alvo para predição
        """
        print("📊 Preparando dados para treinamento...")
        
        if self.df is None:
            raise ValueError("Execute engenharia_features primeiro!")
        
        # Selecionar features numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target not in self.df.columns:
            raise ValueError(f"Target '{target}' não encontrado!")
        
        features_all = [c for c in numeric_cols if c != target]
        
        # Garantir colunas importantes
        important_expected = ['Custo Total', 'Frete', 'Custo Combustível', 'Custo Manutenção', 'Custo Motorista']
        for col in important_expected:
            if col not in features_all and col in self.df.columns:
                features_all.append(col)
        
        features_all = sorted(list(set(features_all)))
        
        # Construir X e y
        X = self.df[features_all].copy()
        y = self.df[target].copy()
        
        # Preencher valores ausentes
        X = X.fillna(X.mean())
        
        # Split treino/teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Salvar lista de features
        pd.Series(features_all, name='feature').to_csv(
            os.path.join(self.output_path, 'features_list.csv'), index=False
        )
        
        print("✅ Dados preparados:")
        print(f"   Treino: {len(self.X_train)} amostras")
        print(f"   Teste:  {len(self.X_test)} amostras")
        print(f"   Features usadas: {len(features_all)}")
        print(f"📁 Lista de features salvo em {os.path.join(self.output_path, 'features_list.csv')}")
    
    def treinar_modelos(self):
        """Treina múltiplos modelos e seleciona o melhor."""
        print("🤖 Iniciando treinamento de modelos...")
        
        modelos = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=5)
        }
        
        print("=" * 80)
        print("TREINAMENTO DE MODELOS")
        print("=" * 80)
        
        for nome, modelo in modelos.items():
            print(f"🔄 Treinando {nome}...")
            modelo.fit(self.X_train, self.y_train)
            
            y_pred_train = modelo.predict(self.X_train)
            y_pred_test = modelo.predict(self.X_test)
            
            r2_train = r2_score(self.y_train, y_pred_train)
            r2_test = r2_score(self.y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            mae_test = mean_absolute_error(self.y_test, y_pred_test)
            
            self.resultados[nome] = {
                'modelo': modelo,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse': rmse_test,
                'mae': mae_test,
                'y_pred': y_pred_test
            }
            
            print(f"   ✅ R² (treino): {r2_train:.4f}")
            print(f"   ✅ R² (teste): {r2_test:.4f}")
            print(f"   ✅ RMSE: {rmse_test:.4f}")
            print(f"   ✅ MAE: {mae_test:.4f}")
        
        # Identificar melhor modelo
        self.melhor_modelo_nome = max(self.resultados, key=lambda x: self.resultados[x]['r2_test'])
        self.melhor_modelo = self.resultados[self.melhor_modelo_nome]['modelo']
        
        print(f"🏆 Melhor modelo: {self.melhor_modelo_nome} (R² teste = {self.resultados[self.melhor_modelo_nome]['r2_test']:.4f})")
        
        # Salvar métricas
        self._salvar_metricas()
        self._salvar_modelo()
    
    def _salvar_metricas(self):
        """Salva métricas dos modelos."""
        metrics_df = pd.DataFrame([
            {
                'Model': name, 
                'R²_Train': self.resultados[name]['r2_train'], 
                'R²_Test': self.resultados[name]['r2_test'],
                'RMSE_Test': self.resultados[name]['rmse'], 
                'MAE_Test': self.resultados[name]['mae']
            }
            for name in self.resultados
        ])
        metrics_df.to_csv(os.path.join(self.output_path, 'model_metrics_summary.csv'), index=False)
        print(f"📊 Resumo de métricas salvo em {os.path.join(self.output_path, 'model_metrics_summary.csv')}")
    
    def _salvar_modelo(self):
        """Salva o melhor modelo."""
        modelo_path = os.path.join(self.output_path, 'modelos', 'melhor_modelo.pkl')
        joblib.dump(self.melhor_modelo, modelo_path)
        print(f"💾 Melhor modelo salvo em {modelo_path}")
    
    def gerar_graficos(self):
        """Gera gráficos comparativos dos modelos."""
        print("📈 Gerando gráficos comparativos...")
        
        import matplotlib.ticker as mtick
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Gráfico 1: Comparação R²
        ax1 = axes[0, 0]
        modelos_nomes = list(self.resultados.keys())
        r2_scores = [self.resultados[m]['r2_test'] for m in modelos_nomes]
        colors = ['#3498DB', '#2ECC71', '#E74C3C']
        bars = ax1.bar(modelos_nomes, r2_scores, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('R² (teste)')
        ax1.set_title('Comparação de Performance (R²)')
        ax1.set_ylim(min(-0.5, min(r2_scores)-0.05), max(1.0, max(r2_scores)+0.05))
        for bar, score in zip(bars, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2., score + 0.01, f'{score:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 2: Predito vs Real
        ax2 = axes[0, 1]
        melhor_pred = self.resultados[self.melhor_modelo_nome]['y_pred']
        ax2.scatter(self.y_test, melhor_pred, alpha=0.6, s=40, color='#3498DB', edgecolors='black')
        mn, mx = min(self.y_test.min(), melhor_pred.min()), max(self.y_test.max(), melhor_pred.max())
        ax2.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Predição Perfeita')
        ax2.set_xlabel('Margem Real (%)')
        ax2.set_ylabel('Margem Predita (%)')
        ax2.set_title(f'Predito vs Real - {self.melhor_modelo_nome}')
        ax2.legend()
        
        # Gráfico 3: Distribuição de erros
        ax3 = axes[1, 0]
        erros = self.y_test - melhor_pred
        ax3.hist(erros, bins=30, color='#9B59B6', alpha=0.8, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Erro de Predição (%)')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Distribuição dos Erros (Melhor Modelo)')
        
        # Gráfico 4: Importância das features
        ax4 = axes[1, 1]
        self._plotar_importancia_features(ax4)
        
        plt.tight_layout()
        grafico_path = os.path.join(self.output_path, 'graficos', '05_importancia_features.png')
        plt.savefig(grafico_path, dpi=250, bbox_inches='tight')
        print(f"📊 Gráfico de importância salvo em {grafico_path}")
        plt.close()
        
        # Salvar importâncias detalhadas
        self._salvar_importancias_detalhadas()
    
    def _plotar_importancia_features(self, ax):
        """Plota importância das features para o melhor modelo."""
        if hasattr(self.melhor_modelo, 'feature_importances_'):
            imp = self.melhor_modelo.feature_importances_
            idx = np.argsort(imp)[-10:]  # Top 10
            ax.barh(range(len(idx)), imp[idx], color='#F39C12', alpha=0.8, edgecolor='black')
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels([self.X_train.columns[i] for i in idx])
            ax.set_title(f'Top 10 Importâncias ({self.melhor_modelo_nome})')
        elif hasattr(self.melhor_modelo, 'coef_'):
            coefs = self.melhor_modelo.coef_
            idx = np.argsort(np.abs(coefs))[-10:]
            ax.barh(range(len(idx)), np.abs(coefs[idx]), color='#F39C12', alpha=0.8, edgecolor='black')
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels([self.X_train.columns[i] for i in idx])
            ax.set_title('Top 10 Importâncias (Coef. Abs) — Linear')
        else:
            ax.text(0.5, 0.5, 'Importância de features não disponível', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _salvar_importancias_detalhadas(self):
        """Salva importâncias detalhadas por modelo."""
        def get_feature_importance_df(model, features, model_name):
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                df_imp = pd.DataFrame({'feature': features, 'importance': imp})
                df_imp = df_imp.sort_values('importance', ascending=False)
                df_imp['model'] = model_name
                return df_imp
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                df_imp = pd.DataFrame({'feature': features, 'importance': np.abs(coef)})
                df_imp = df_imp.sort_values('importance', ascending=False)
                df_imp['model'] = model_name
                return df_imp
            else:
                return pd.DataFrame()
        
        all_imps = []
        for name in self.resultados:
            dfimp = get_feature_importance_df(self.resultados[name]['modelo'], self.X_train.columns, name)
            if not dfimp.empty:
                all_imps.append(dfimp)
        
        if all_imps:
            feat_imps_df = pd.concat(all_imps, axis=0)
            feat_imps_df.to_csv(os.path.join(self.output_path, 'feature_importances_by_model.csv'), index=False)
            print(f"📊 Importâncias detalhadas salvas em {os.path.join(self.output_path, 'feature_importances_by_model.csv')}")
    
    def gerar_insights_avancados(self, dados_mensais=None):
        """
        Gera insights avançados, cenários e recomendações.
        
        Args:
            dados_mensais (pd.DataFrame): Dados mensais para referência
            
        Returns:
            dict: Resultados completos da análise
        """
        print("🔍 Gerando insights avançados e cenários...")
        
        print("=" * 80)
        print("INSIGHTS DO MODELO PREDITIVO — CENÁRIOS, PROJEÇÕES E RECOMENDAÇÕES")
        print("=" * 80)
        
        # Construir dataframe de importâncias
        if hasattr(self.melhor_modelo, 'feature_importances_'):
            imp = self.melhor_modelo.feature_importances_
            self.feat_imp_df = pd.DataFrame({
                'Feature': self.X_train.columns, 
                'Importância': imp
            }).sort_values('Importância', ascending=False).reset_index(drop=True)
        else:
            if hasattr(self.melhor_modelo, 'coef_'):
                coefs = self.melhor_modelo.coef_
                self.feat_imp_df = pd.DataFrame({
                    'Feature': self.X_train.columns, 
                    'Coeficiente': coefs, 
                    'Importância Abs': np.abs(coefs)
                }).sort_values('Importância Abs', ascending=False).reset_index(drop=True)
            else:
                raise RuntimeError("Melhor modelo não expõe importances nem coef_.")
        
        print("\n🎯 TOP 15 FEATURES (ordenadas por importância):")
        print(self.feat_imp_df.head(15))
        
        # Gerar cenários
        cenarios_df = self._gerar_cenarios(dados_mensais)
        
        # Gerar gráficos de cenários
        self._gerar_graficos_cenarios(cenarios_df)
        
        # Gerar simulações de ações
        acoes_df = self._simular_acoes(dados_mensais)
        
        # Gerar recomendações
        self._gerar_recomendacoes(cenarios_df, acoes_df)
        
        return {
            'cenarios': cenarios_df,
            'acoes': acoes_df,
            'importancias': self.feat_imp_df
        }
    
    def _gerar_cenarios(self, dados_mensais):
        """Gera cenários de otimização."""
        cenario_base = self.X_test.mean()
        cenario_base = cenario_base.reindex(self.X_train.columns)
        
        # Definir função auxiliar para prever
        def predict_series(s):
            s_df = s.to_frame().T
            s_df = s_df[self.X_train.columns]
            return self.melhor_modelo.predict(s_df)[0]

        margem_atual = predict_series(cenario_base)

        # Função para ajustar Custo Total após mudar componentes
        def recompute_custo_total(s):
            s = s.copy()
            if 'Custo Total' in s.index:
                comps = ['Custo Combustível', 'Custo Manutenção', 'Custo Motorista']
                sum_comps = 0.0
                for c in comps:
                    if c in s.index:
                        sum_comps += float(s[c])
                try:
                    orig_total = float(self.X_test.mean().get('Custo Total', s.get('Custo Total', np.nan)))
                    orig_components_sum = 0.0
                    for c in comps:
                        if c in self.X_test.columns:
                            orig_components_sum += float(self.X_test[c].mean())
                    residual = orig_total - orig_components_sum
                    new_total = sum_comps + residual
                    s['Custo Total'] = new_total
                except Exception:
                    pass
            return s

        # Definir os 4 cenários
        cenarios = {}

        # Cenario 1 — atual (base)
        cenarios['Atual'] = cenario_base.copy()

        # Cenario 2 — Otimização de Custos
        c2 = cenario_base.copy()
        if 'Custo Combustível' in c2.index:
            c2['Custo Combustível'] *= 0.9
        if 'Custo Manutenção' in c2.index:
            c2['Custo Manutenção'] *= 0.9
        if 'Custo Motorista' in c2.index:
            c2['Custo Motorista'] *= 0.95
        c2 = recompute_custo_total(c2)
        cenarios['Otimização Custos'] = c2

        # Cenario 3 — Eficiência Operacional
        c3 = cenario_base.copy()
        if 'Entregas' in c3.index:
            c3['Entregas'] *= 1.15
        if 'KM Percorridos' in c3.index:
            c3['KM Percorridos'] *= 0.9
        c3 = recompute_custo_total(c3)
        cenarios['Eficiência Operacional'] = c3

        # Cenario 4 — Combinado
        c4 = cenario_base.copy()
        if 'Custo Combustível' in c4.index:
            c4['Custo Combustível'] *= 0.9
        if 'Custo Manutenção' in c4.index:
            c4['Custo Manutenção'] *= 0.9
        if 'Custo Motorista' in c4.index:
            c4['Custo Motorista'] *= 0.95
        if 'Entregas' in c4.index:
            c4['Entregas'] *= 1.15
        if 'KM Percorridos' in c4.index:
            c4['KM Percorridos'] *= 0.9
        c4 = recompute_custo_total(c4)
        cenarios['Combinado'] = c4

        # Calcular métricas dos cenários
        monthly_avg_cost = self._calcular_custo_mensal_medio(dados_mensais)
        summary_rows = []

        for name, s in cenarios.items():
            margem = predict_series(s)
            delta_pp = margem - margem_atual
            
            custo_total_base = float(cenario_base['Custo Total']) if 'Custo Total' in cenario_base.index else np.nan
            custo_total_new = float(s['Custo Total']) if 'Custo Total' in s.index else np.nan
            custo_delta = custo_total_base - custo_total_new if not (np.isnan(custo_total_base) or np.isnan(custo_total_new)) else np.nan
            
            economia_mensal = (custo_delta / custo_total_base * monthly_avg_cost) if (not np.isnan(custo_delta) and custo_total_base != 0 and not np.isnan(custo_total_base)) else np.nan
            economia_anual = economia_mensal * 12 if not np.isnan(economia_mensal) else np.nan

            summary_rows.append({
                'Cenário': name,
                'Margem Prevista (%)': margem,
                'Delta (p.p.)': delta_pp,
                'Custo Total Base (R$/dia)': custo_total_base,
                'Custo Total Novo (R$/dia)': custo_total_new,
                'Economia por dia (R$)': custo_delta,
                'Economia mensal estimada (R$)': economia_mensal,
                'Economia anual estimada (R$)': economia_anual
            })

        cenarios_df = pd.DataFrame(summary_rows).set_index('Cenário')
        cenarios_df = cenarios_df.loc[['Atual', 'Otimização Custos', 'Eficiência Operacional', 'Combinado']]

        print("\n📈 Resumo dos Cenários:")
        print(cenarios_df)

        cenarios_df.to_csv(os.path.join(self.output_path, 'cenarios_detalhados_comparativo.csv'))
        return cenarios_df

    def _calcular_custo_mensal_medio(self, dados_mensais):
        """Calcula custo mensal médio para estimativas."""
        if dados_mensais is not None and 'Custo Total' in dados_mensais.columns:
            return float(dados_mensais['Custo Total'].mean())
        elif 'Custo Total' in self.X_test.columns:
            return float(self.X_test['Custo Total'].mean()) * 30
        else:
            return np.nan

    def _gerar_graficos_cenarios(self, cenarios_df):
        """Gera gráficos dos cenários."""
        # Gráfico de margens por cenário
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=cenarios_df.index, y=cenarios_df['Margem Prevista (%)'].values, palette='Blues_r')
        ax.set_title('Margem Prevista por Cenário (%)')
        ax.set_ylabel('Margem (%)')
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width()/2., p.get_height()), 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.savefig(os.path.join(self.output_path, 'graficos', '06_previsoes_modelo.png'), 
                   dpi=250, bbox_inches='tight')
        plt.close()
        
        # Gráfico de economia mensal
        plt.figure(figsize=(10, 6))
        econ = cenarios_df['Economia mensal estimada (R$)'].fillna(0)
        ax2 = sns.barplot(x=econ.index, y=econ.values, palette='Greens')
        ax2.set_title('Economia Mensal Estimada por Cenário (R$)')
        ax2.set_ylabel('Economia mensal (R$)')
        for p in ax2.patches:
            ax2.annotate(f"R$ {p.get_height():,.0f}", (p.get_x() + p.get_width()/2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.savefig(os.path.join(self.output_path, 'graficos', '07_dashboard_executivo.png'), 
                   dpi=250, bbox_inches='tight')
        plt.close()

    def _simular_acoes(self, dados_mensais):
        """Simula impacto de ações específicas por feature."""
        cenario_base = self.X_test.mean()
        cenario_base = cenario_base.reindex(self.X_train.columns)
        
        # Definir função auxiliar para prever
        def predict_series(s):
            s_df = s.to_frame().T
            s_df = s_df[self.X_train.columns]
            return self.melhor_modelo.predict(s_df)[0]

        margem_atual = predict_series(cenario_base)
        monthly_avg_cost = self._calcular_custo_mensal_medio(dados_mensais)
        
        action_map = {
            'Custo Combustível': {'mult': 0.90, 'label': '-10% Custo Combustível'},
            'Custo Manutenção': {'mult': 0.90, 'label': '-10% Custo Manutenção'},
            'Custo Motorista': {'mult': 0.95, 'label': '-5% Custo Motorista'},
            'Entregas': {'mult': 1.15, 'label': '+15% Entregas'},
            'KM Percorridos': {'mult': 0.90, 'label': '-10% KM Percorridos'},
        }
        
        top_features = self.feat_imp_df.head(10)['Feature'].tolist()
        action_results = []
        
        for feat in top_features:
            if feat in action_map:
                s = cenario_base.copy()
                info = action_map[feat]
                s[feat] = s[feat] * info['mult']
                
                # Recalcular custo total se necessário
                if any(custo in feat for custo in ['Custo', 'Combustível', 'Manutenção', 'Motorista']):
                    s = self._recompute_custo_total(s)
                
                new_marg = predict_series(s)
                delta_pp = new_marg - margem_atual
                
                custo_base = float(cenario_base['Custo Total']) if 'Custo Total' in cenario_base.index else np.nan
                custo_new = float(s['Custo Total']) if 'Custo Total' in s.index else np.nan
                custo_delta = custo_base - custo_new if not (np.isnan(custo_base) or np.isnan(custo_new)) else np.nan
                
                economia_mensal = (custo_delta / custo_base * monthly_avg_cost) if (not np.isnan(custo_delta) and custo_base != 0 and not np.isnan(custo_base)) else np.nan
                
                action_results.append({
                    'Feature': feat,
                    'Ação proposta': info['label'],
                    'Margem nova (%)': new_marg,
                    'Delta (p.p.)': delta_pp,
                    'Economia mensal estimada (R$)': economia_mensal
                })
        
        actions_df = pd.DataFrame(action_results).sort_values('Delta (p.p.)', ascending=False)
        actions_df.to_csv(os.path.join(self.output_path, 'simulacao_acoes_por_feature.csv'), index=False)
        
        return actions_df

    def _recompute_custo_total(self, s):
        """Recalcula custo total após alterações nos componentes."""
        s = s.copy()
        if 'Custo Total' in s.index:
            comps = ['Custo Combustível', 'Custo Manutenção', 'Custo Motorista']
            sum_comps = 0.0
            for c in comps:
                if c in s.index:
                    sum_comps += float(s[c])
            try:
                orig_total = float(self.X_test.mean().get('Custo Total'))
                orig_components_sum = 0.0
                for c in comps:
                    if c in self.X_test.columns:
                        orig_components_sum += float(self.X_test[c].mean())
                residual = orig_total - orig_components_sum
                new_total = sum_comps + residual
                s['Custo Total'] = new_total
            except Exception:
                pass
        return s

    def _gerar_recomendacoes(self, cenarios_df, acoes_df):
        """Gera recomendações baseadas nos resultados."""
        print("\n" + "="*80)
        print("RECOMENDAÇÕES PRÁTICAS (priorizadas)")
        print("="*80)
        
        # Prioridade 1: Otimização de Custos
        print("\n1) Prioridade: Otimização de Custos")
        cenario_otim = cenarios_df.loc['Otimização Custos']
        print(f"   → Ganho estimado: {cenario_otim['Margem Prevista (%)']:.2f}% (delta {cenario_otim['Delta (p.p.)']:.2f} p.p.)")
        print(f"   → Economia mensal estimada: R$ {cenario_otim['Economia mensal estimada (R$)']:.0f}")
        
        # Prioridade 2: Eficiência Operacional
        print("\n2) Prioridade: Eficiência Operacional")
        cenario_efic = cenarios_df.loc['Eficiência Operacional']
        print(f"   → Ganho estimado: {cenario_efic['Margem Prevista (%)']:.2f}% (delta {cenario_efic['Delta (p.p.)']:.2f} p.p.)")
        print(f"   → Economia mensal estimada: R$ {cenario_efic['Economia mensal estimada (R$)']:.0f}")
        
        # Top 3 ações específicas
        print("\n3) Ações Específicas Recomendadas:")
        top_acoes = acoes_df.head(3)
        for _, acao in top_acoes.iterrows():
            print(f"   → {acao['Ação proposta']}: +{acao['Delta (p.p.)']:.2f} p.p. na margem")

    def gerar_relatorio_completo(self, dados_mensais=None):
        """
        Executa análise preditiva completa.
        
        Args:
            dados_mensais (pd.DataFrame): Dados mensais para referência
            
        Returns:
            dict: Resultados completos da análise
        """
        print("🚀 Iniciando análise preditiva completa...")
        
        # Pipeline completo
        self.engenharia_features(dados_mensais=dados_mensais)
        self.preparar_dados()
        self.treinar_modelos()
        self.gerar_graficos()
        insights = self.gerar_insights_avancados(dados_mensais=dados_mensais)
        
        # Compilar resultados
        resultados = {
            'metricas': pd.read_csv(os.path.join(self.output_path, 'model_metrics_summary.csv')),
            'melhor_modelo': self.melhor_modelo_nome,
            'melhor_r2': self.resultados[self.melhor_modelo_nome]['r2_test'],
            'cenarios': insights['cenarios'],
            'acoes_recomendadas': insights['acoes'],
            'importancias': insights['importancias']
        }
        
        print("✅ Análise preditiva concluída com sucesso!")
        return resultados


# Função auxiliar para uso direto
def executar_analise_preditiva(dados_logistica, dados_mensais=None, output_path='output'):
    """
    Função conveniente para executar análise preditiva.
    
    Args:
        dados_logistica (pd.DataFrame): Dados diários de logística
        dados_mensais (pd.DataFrame): Dados mensais para referência
        output_path (str): Pasta de saída
        
    Returns:
        dict: Resultados da análise
    """
    analise = AnalisePreditiva(dados_logistica, output_path)
    return analise.gerar_relatorio_completo(dados_mensais=dados_mensais)


if __name__ == "__main__":
    # Exemplo de uso direto
    print("🔧 Testando módulo de análise preditiva...")
    
    # Carregar dados de exemplo (substitua por seus dados)
    try:
        dados_logistica = pd.read_csv('data/dados_logistica.csv')
        dados_logistica['Data'] = pd.to_datetime(dados_logistica['Data'])
        
        # Executar análise
        resultados = executar_analise_preditiva(dados_logistica)
        print("✅ Teste concluído com sucesso!")
        
    except FileNotFoundError:
        print("❌ Arquivos de dados não encontrados. Execute a partir do main.py.")