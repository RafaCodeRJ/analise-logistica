"""
Módulo para análise exploratória de dados logísticos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import warnings

class AnaliseExploratoria:
    """Classe responsável pela análise exploratória dos dados."""
    
    def __init__(self, dados_logistica, dados_mensais=None, tabela_acoes=None, output_path='output'):
        """
        Inicializa a análise exploratória.
        
        Args:
            dados_logistica (pd.DataFrame): Dados de logística diários
            dados_mensais (pd.DataFrame): Dados mensais consolidados
            tabela_acoes (pd.DataFrame): Tabela de ações propostas
            output_path (str): Caminho para salvar os resultados
        """
        self.dados_logistica = dados_logistica
        self.dados_mensais = dados_mensais
        self.tabela_acoes = tabela_acoes
        self.output_path = output_path
        
        # Criar pastas de output
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'graficos'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'relatorios'), exist_ok=True)
        
        # Configurações
        warnings.filterwarnings('ignore')
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (14, 6)
        plt.rcParams['font.size'] = 10
        
        # Configuração de exibição do pandas
        pd.set_option('display.max_columns', None)
        pd.set_option('display.precision', 2)
        
        print("✅ Análise Exploratória inicializada!")
    
    def gerar_relatorio_completo(self):
        """
        Gera relatório completo de análise exploratória.
        
        Returns:
            dict: Resultados completos da análise
        """
        print("📊 Iniciando análise exploratória completa...")
        
        resultados = {
            'estrutura_dados': self._analisar_estrutura_dados(),
            'estatisticas_descritivas': self._gerar_estatisticas_descritivas(),
            'composicao_custos': self._analisar_composicao_custos(),
            'rentabilidade': self._analisar_rentabilidade(),
            'tendencias_temporais': self._analisar_tendencias_temporais(),
            'eficiencia_operacional': self._analisar_eficiencia_operacional(),  # MÉTODO ADICIONADO
            'dias_criticos': self._identificar_dias_criticos(),
            'plano_acoes': self._analisar_plano_acoes() if self.tabela_acoes is not None else None,
            'resumo_executivo': self._gerar_resumo_executivo()
        }
        
        # Salvar relatório completo
        self._salvar_relatorio_completo(resultados)
        
        print("✅ Análise exploratória concluída com sucesso!")
        return resultados
    
    def _analisar_estrutura_dados(self):
        """Analisa a estrutura dos dados."""
        print("   📋 Analisando estrutura dos dados...")
        
        info_buffer = []
        
        # Informações básicas
        info_buffer.append("=" * 80)
        info_buffer.append("ESTRUTURA DOS DADOS LOGÍSTICOS")
        info_buffer.append("=" * 80)
        
        # Usar StringIO para capturar o output do info()
        import io
        buffer = io.StringIO()
        self.dados_logistica.info(buf=buffer)
        info_buffer.append(buffer.getvalue())
        
        info_buffer.append("\n" + "=" * 80)
        info_buffer.append("PRIMEIRAS LINHAS")
        info_buffer.append("=" * 80)
        info_buffer.append(str(self.dados_logistica.head(10)))
        
        # Informações do período
        if 'Data' in self.dados_logistica.columns:
            info_buffer.append(f"\n📅 Período analisado: {self.dados_logistica['Data'].min().strftime('%d/%m/%Y')} a {self.dados_logistica['Data'].max().strftime('%d/%m/%Y')}")
        
        info_buffer.append(f"📊 Total de registros diários: {len(self.dados_logistica)}")
        
        if self.dados_mensais is not None:
            info_buffer.append(f"📊 Total de meses: {len(self.dados_mensais)}")
        
        if self.tabela_acoes is not None:
            info_buffer.append(f"📋 Total de ações planejadas: {len(self.tabela_acoes)}")
        
        return "\n".join(info_buffer)
    
    def _gerar_estatisticas_descritivas(self):
        """Gera estatísticas descritivas completas."""
        print("   📈 Gerando estatísticas descritivas...")
        
        stats_buffer = []
        stats_buffer.append("=" * 80)
        stats_buffer.append("ESTATÍSTICAS DESCRITIVAS")
        stats_buffer.append("=" * 80)
        
        # Estatísticas básicas
        stats = self.dados_logistica.describe()
        stats_buffer.append(str(stats.round(2)))
        
        stats_buffer.append("\n" + "=" * 80)
        stats_buffer.append("INFORMAÇÕES ADICIONAIS")
        stats_buffer.append("=" * 80)
        
        # Cálculos específicos
        if 'Custo Total' in self.dados_logistica.columns:
            stats_buffer.append(f"💰 CUSTOS:")
            stats_buffer.append(f"   Custo Total Médio/Dia: R$ {self.dados_logistica['Custo Total'].mean():,.2f}")
            stats_buffer.append(f"   Custo Total Mediano/Dia: R$ {self.dados_logistica['Custo Total'].median():,.2f}")
            stats_buffer.append(f"   Desvio Padrão: R$ {self.dados_logistica['Custo Total'].std():,.2f}")
        
        if 'Frete' in self.dados_logistica.columns:
            stats_buffer.append(f"📊 RECEITAS:")
            stats_buffer.append(f"   Frete Médio/Dia: R$ {self.dados_logistica['Frete'].mean():,.2f}")
            stats_buffer.append(f"   Frete Mediano/Dia: R$ {self.dados_logistica['Frete'].median():,.2f}")
        
        if 'Margem %' in self.dados_logistica.columns:
            stats_buffer.append(f"💹 MARGENS:")
            stats_buffer.append(f"   Margem Média: {self.dados_logistica['Margem %'].mean():.2f}%")
            stats_buffer.append(f"   Margem Mediana: {self.dados_logistica['Margem %'].median():.2f}%")
            stats_buffer.append(f"   Margem Mínima: {self.dados_logistica['Margem %'].min():.2f}%")
            stats_buffer.append(f"   Margem Máxima: {self.dados_logistica['Margem %'].max():.2f}%")
        
        return "\n".join(stats_buffer)
    
    def _analisar_composicao_custos(self):
        """Analisa a composição dos custos operacionais."""
        print("   💰 Analisando composição de custos...")
        
        # Verificar se as colunas necessárias existem
        colunas_necessarias = ['Custo Combustível', 'Custo Manutenção', 'Custo Motorista', 'Custo Total']
        if not all(col in self.dados_logistica.columns for col in colunas_necessarias):
            return "❌ Colunas de custo não encontradas para análise"
        
        custo_combustivel_total = self.dados_logistica['Custo Combustível'].sum()
        custo_manutencao_total = self.dados_logistica['Custo Manutenção'].sum()
        custo_motorista_total = self.dados_logistica['Custo Motorista'].sum()
        custo_total = self.dados_logistica['Custo Total'].sum()
        
        # Gerar relatório textual
        relatorio = []
        relatorio.append("=" * 80)
        relatorio.append("COMPOSIÇÃO DOS CUSTOS")
        relatorio.append("=" * 80)
        
        relatorio.append(f"💵 Custo Combustível: R$ {custo_combustivel_total:,.2f} ({custo_combustivel_total/custo_total*100:.1f}%)")
        relatorio.append(f"🔧 Custo Manutenção: R$ {custo_manutencao_total:,.2f} ({custo_manutencao_total/custo_total*100:.1f}%)")
        relatorio.append(f"👨‍✈️ Custo Motorista: R$ {custo_motorista_total:,.2f} ({custo_motorista_total/custo_total*100:.1f}%)")
        relatorio.append(f"💰 CUSTO TOTAL: R$ {custo_total:,.2f}")
        
        # Gerar gráficos
        self._gerar_graficos_composicao_custos(
            custo_combustivel_total, 
            custo_manutencao_total, 
            custo_motorista_total
        )
        
        return "\n".join(relatorio)
    
    def _gerar_graficos_composicao_custos(self, custo_combustivel, custo_manutencao, custo_motorista):
        """Gera gráficos da composição de custos."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico de pizza - Composição de custos
        custos = [custo_combustivel, custo_manutencao, custo_motorista]
        labels = ['Combustível', 'Manutenção', 'Motorista']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        explode = (0.05, 0, 0)
        
        ax1.pie(custos, labels=labels, autopct='%1.1f%%', startangle=90, 
                colors=colors, explode=explode, shadow=True)
        ax1.set_title('Composição dos Custos Operacionais', fontsize=14, fontweight='bold')
        
        # Gráfico de barras - Custos médios diários
        custos_medios = {
            'Combustível': self.dados_logistica['Custo Combustível'].mean(),
            'Manutenção': self.dados_logistica['Custo Manutenção'].mean(),
            'Motorista': self.dados_logistica['Custo Motorista'].mean()
        }
        
        ax2.bar(custos_medios.keys(), custos_medios.values(), color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Custos Médios Diários', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Valor (R$)', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, (k, v) in enumerate(custos_medios.items()):
            ax2.text(i, v + 50, f'R$ {v:,.0f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'graficos', '01_composicao_custos.png'), 
                   dpi=250, bbox_inches='tight')
        plt.close()
    
    def _analisar_rentabilidade(self):
        """Analisa a rentabilidade dos dados."""
        print("   💹 Analisando rentabilidade...")
        
        if 'Margem %' not in self.dados_logistica.columns or 'Margem' not in self.dados_logistica.columns:
            return "❌ Colunas de margem não encontradas para análise"
        
        relatorio = []
        relatorio.append("=" * 80)
        relatorio.append("ANÁLISE DE RENTABILIDADE")
        relatorio.append("=" * 80)
        
        margem_total = self.dados_logistica['Margem'].sum()
        frete_total = self.dados_logistica['Frete'].sum()
        
        relatorio.append(f"💰 Frete Total: R$ {frete_total:,.2f}")
        relatorio.append(f"💹 Margem Total: R$ {margem_total:,.2f}")
        relatorio.append(f"📊 Margem Percentual Média: {self.dados_logistica['Margem %'].mean():.2f}%")
        
        # Classificação de dias por rentabilidade
        dias_excelente = len(self.dados_logistica[self.dados_logistica['Margem %'] >= 35])
        dias_bom = len(self.dados_logistica[(self.dados_logistica['Margem %'] >= 25) & (self.dados_logistica['Margem %'] < 35)])
        dias_regular = len(self.dados_logistica[(self.dados_logistica['Margem %'] >= 20) & (self.dados_logistica['Margem %'] < 25)])
        dias_critico = len(self.dados_logistica[self.dados_logistica['Margem %'] < 20])
        
        relatorio.append(f"📈 CLASSIFICAÇÃO DOS DIAS:")
        relatorio.append(f"   🟢 Excelente (≥35%): {dias_excelente} dias ({dias_excelente/len(self.dados_logistica)*100:.1f}%)")
        relatorio.append(f"   🔵 Bom (25-35%): {dias_bom} dias ({dias_bom/len(self.dados_logistica)*100:.1f}%)")
        relatorio.append(f"   🟡 Regular (20-25%): {dias_regular} dias ({dias_regular/len(self.dados_logistica)*100:.1f}%)")
        relatorio.append(f"   🔴 Crítico (<20%): {dias_critico} dias ({dias_critico/len(self.dados_logistica)*100:.1f}%)")
        
        # Gerar gráficos de rentabilidade
        self._gerar_graficos_rentabilidade(dias_excelente, dias_bom, dias_regular, dias_critico)
        
        return "\n".join(relatorio)
    
    def _gerar_graficos_rentabilidade(self, dias_excelente, dias_bom, dias_regular, dias_critico):
        """Gera gráficos de análise de rentabilidade."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Distribuição de margens
        ax1.hist(self.dados_logistica['Margem %'], bins=30, color='#45B7D1', alpha=0.7, edgecolor='black')
        ax1.axvline(self.dados_logistica['Margem %'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Média: {self.dados_logistica["Margem %"].mean():.1f}%')
        ax1.axvline(self.dados_logistica['Margem %'].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Mediana: {self.dados_logistica["Margem %"].median():.1f}%')
        ax1.set_xlabel('Margem (%)', fontsize=12)
        ax1.set_ylabel('Frequência', fontsize=12)
        ax1.set_title('Distribuição das Margens', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Pizza - Classificação de dias
        classificacao = [dias_excelente, dias_bom, dias_regular, dias_critico]
        labels_class = ['Excelente (≥35%)', 'Bom (25-35%)', 'Regular (20-25%)', 'Crítico (<20%)']
        colors_class = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
        
        ax2.pie(classificacao, labels=labels_class, autopct='%1.1f%%', startangle=90,
                colors=colors_class, shadow=True)
        ax2.set_title('Classificação dos Dias por Rentabilidade', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'graficos', '02_rentabilidade.png'),  # NOME CORRIGIDO
                   dpi=250, bbox_inches='tight')
        plt.close()
    
    def _analisar_tendencias_temporais(self):
        """Analisa tendências temporais nos dados."""
        print("   📅 Analisando tendências temporais...")
        
        if 'Data' not in self.dados_logistica.columns:
            return "❌ Coluna 'Data' não encontrada para análise temporal"
        
        # Ordenar por data
        dados_sorted = self.dados_logistica.sort_values('Data')
        
        # Criar figura com subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Gráfico 1: Evolução da Margem
        if 'Margem %' in self.dados_logistica.columns:
            axes[0].plot(dados_sorted['Data'], dados_sorted['Margem %'], 
                        color='#2ECC71', linewidth=1.5, alpha=0.7)
            axes[0].axhline(y=self.dados_logistica['Margem %'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Média: {self.dados_logistica["Margem %"].mean():.1f}%')
            axes[0].axhline(y=20, color='orange', linestyle='--', linewidth=2, label='Limite Crítico: 20%')
            axes[0].fill_between(dados_sorted['Data'], dados_sorted['Margem %'], 
                               alpha=0.3, color='#2ECC71')
            axes[0].set_ylabel('Margem (%)', fontsize=12, fontweight='bold')
            axes[0].set_title('Evolução da Margem de Lucro', fontsize=14, fontweight='bold')
            axes[0].legend(loc='best')
            axes[0].grid(alpha=0.3)
        
        # Gráfico 2: Evolução dos Custos
        if 'Custo Total' in self.dados_logistica.columns and 'Frete' in self.dados_logistica.columns:
            axes[1].plot(dados_sorted['Data'], dados_sorted['Custo Total'], 
                        label='Custo Total', color='#E74C3C', linewidth=2)
            axes[1].plot(dados_sorted['Data'], dados_sorted['Frete'], 
                        label='Frete', color='#3498DB', linewidth=2)
            axes[1].fill_between(dados_sorted['Data'], dados_sorted['Custo Total'],
                               dados_sorted['Frete'], alpha=0.3, color='#2ECC71', 
                               label='Margem')
            axes[1].set_ylabel('Valor (R$)', fontsize=12, fontweight='bold')
            axes[1].set_title('Evolução de Custos e Receitas', fontsize=14, fontweight='bold')
            axes[1].legend(loc='best')
            axes[1].grid(alpha=0.3)
        
        # Gráfico 3: Custo por KM
        if 'Custo/KM' in self.dados_logistica.columns:
            axes[2].plot(dados_sorted['Data'], dados_sorted['Custo/KM'], 
                        color='#9B59B6', linewidth=1.5)
            axes[2].axhline(y=self.dados_logistica['Custo/KM'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Média: R$ {self.dados_logistica["Custo/KM"].mean():.4f}')
            axes[2].fill_between(dados_sorted['Data'], dados_sorted['Custo/KM'], 
                               alpha=0.3, color='#9B59B6')
            axes[2].set_xlabel('Data', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Custo/KM (R$)', fontsize=12, fontweight='bold')
            axes[2].set_title('Evolução do Custo por Quilômetro', fontsize=14, fontweight='bold')
            axes[2].legend(loc='best')
            axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'graficos', '03_tendencias_temporais.png'),  # NOME CORRIGIDO
                   dpi=250, bbox_inches='tight')
        plt.close()
        
        # Análise mensal
        relatorio = []
        relatorio.append("\n" + "=" * 80)
        relatorio.append("ANÁLISE MENSAL")
        relatorio.append("=" * 80)
        
        if self.dados_mensais is not None:
            dados_mensais_sorted = self.dados_mensais.sort_values('Mês')
            colunas_interesse = ['Mês', 'Custo Total', 'Frete', 'Margem', 'Margem %']
            colunas_disponiveis = [c for c in colunas_interesse if c in self.dados_mensais.columns]
            
            if colunas_disponiveis:
                relatorio.append(dados_mensais_sorted[colunas_disponiveis].to_string(index=False))
        
        return "\n".join(relatorio)
    
    def _analisar_eficiencia_operacional(self):  # MÉTODO ADICIONADO
        """Analisa a eficiência operacional."""
        print("   🚚 Analisando eficiência operacional...")
        
        relatorio = []
        relatorio.append("=" * 80)
        relatorio.append("EFICIÊNCIA OPERACIONAL")
        relatorio.append("=" * 80)
        
        # Análise de entregas
        if 'Entregas' in self.dados_logistica.columns:
            relatorio.append(f"🚚 ENTREGAS:")
            relatorio.append(f"   Total de entregas: {self.dados_logistica['Entregas'].sum():,.0f}")
            relatorio.append(f"   Média de entregas/dia: {self.dados_logistica['Entregas'].mean():.0f}")
            relatorio.append(f"   Mediana de entregas/dia: {self.dados_logistica['Entregas'].median():.0f}")
        
        # Análise de quilometragem
        if 'KM Percorridos' in self.dados_logistica.columns:
            relatorio.append(f"🛣️ QUILOMETRAGEM:")
            relatorio.append(f"   Total de KM percorridos: {self.dados_logistica['KM Percorridos'].sum():,.0f} km")
            relatorio.append(f"   Média de KM/dia: {self.dados_logistica['KM Percorridos'].mean():,.0f} km")
            
            if 'KM/Entrega' in self.dados_logistica.columns:
                relatorio.append(f"   Média de KM/entrega: {self.dados_logistica['KM/Entrega'].mean():.2f} km")
        
        # Análise de carga
        if 'Peso (ton)' in self.dados_logistica.columns:
            relatorio.append(f"📦 CARGA:")
            relatorio.append(f"   Total transportado: {self.dados_logistica['Peso (ton)'].sum():,.2f} toneladas")
            relatorio.append(f"   Média/dia: {self.dados_logistica['Peso (ton)'].mean():.2f} toneladas")
        
        # Análise de correlação
        relatorio.append("\n" + "=" * 80)
        relatorio.append("CORRELAÇÃO ENTRE VARIÁVEIS OPERACIONAIS")
        relatorio.append("=" * 80)
        
        colunas_analise = ['Entregas', 'KM Percorridos', 'Peso (ton)', 'Custo Total', 'Frete', 'Margem %']
        colunas_disponiveis = [c for c in colunas_analise if c in self.dados_logistica.columns]
        
        if len(colunas_disponiveis) >= 2:
            correlacao = self.dados_logistica[colunas_disponiveis].corr()
            
            # Gerar heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlacao, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Matriz de Correlação - Variáveis Operacionais', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'graficos', '04_matriz_correlacao.png'),  # NOME CORRIGIDO
                       dpi=250, bbox_inches='tight')
            plt.close()
            
            # Identificar correlações fortes
            relatorio.append("📊 Principais correlações identificadas:")
            for i in range(len(correlacao.columns)):
                for j in range(i+1, len(correlacao.columns)):
                    corr_value = correlacao.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        relatorio.append(f"   • {correlacao.columns[i]} ↔ {correlacao.columns[j]}: {corr_value:.2f}")
        else:
            relatorio.append("Não há colunas suficientes para análise de correlação.")
        
        return "\n".join(relatorio)
    
    def _identificar_dias_criticos(self):
        """Identifica e analisa dias críticos."""
        print("   🔴 Identificando dias críticos...")
        
        if 'Margem %' not in self.dados_logistica.columns:
            return "❌ Coluna 'Margem %' não encontrada para análise de dias críticos"
        
        relatorio = []
        relatorio.append("=" * 80)
        relatorio.append("DIAS CRÍTICOS (Margem < 20%)")
        relatorio.append("=" * 80)
        
        dias_criticos = self.dados_logistica[self.dados_logistica['Margem %'] < 20].sort_values('Margem %')
        relatorio.append(f"Total de dias críticos: {len(dias_criticos)} ({len(dias_criticos)/len(self.dados_logistica)*100:.1f}% do período)")
        
        if len(dias_criticos) > 0:
            relatorio.append(f"🔴 TOP 10 PIORES DIAS:")
            colunas_interesse = ['Data', 'Margem %', 'Custo Total', 'Frete', 'Entregas']
            if 'KM/Entrega' in self.dados_logistica.columns:
                colunas_interesse.append('KM/Entrega')
            
            colunas_disponiveis = [c for c in colunas_interesse if c in dias_criticos.columns]
            relatorio.append(dias_criticos[colunas_disponiveis].head(10).to_string(index=False))
            
            # Análise dos dias críticos
            relatorio.append(f"📊 CARACTERÍSTICAS DOS DIAS CRÍTICOS:")
            relatorio.append(f"   Margem média: {dias_criticos['Margem %'].mean():.2f}%")
            
            if 'Custo/KM' in dias_criticos.columns:
                relatorio.append(f"   Custo/KM médio: R$ {dias_criticos['Custo/KM'].mean():.4f}")
            
            if 'KM/Entrega' in dias_criticos.columns:
                relatorio.append(f"   KM/Entrega médio: {dias_criticos['KM/Entrega'].mean():.2f} km")
            
            relatorio.append(f"   Entregas médias: {dias_criticos['Entregas'].mean():.0f}")
        
        # Identificar dias com alto custo/km
        if 'Custo/KM' in self.dados_logistica.columns:
            alto_custo_km = self.dados_logistica[self.dados_logistica['Custo/KM'] > self.dados_logistica['Custo/KM'].quantile(0.9)]
            
            relatorio.append(f"⚠️ DIAS COM ALTO CUSTO/KM (Top 10%):")
            relatorio.append(f"   Total de dias: {len(alto_custo_km)}")
            relatorio.append(f"   Custo/KM médio: R$ {alto_custo_km['Custo/KM'].mean():.4f}")
            relatorio.append(f"   vs. Média geral: R$ {self.dados_logistica['Custo/KM'].mean():.4f}")
        
        # Gerar visualizações
        self._gerar_graficos_dias_criticos(dias_criticos)
        
        return "\n".join(relatorio)
    
    def _gerar_graficos_dias_criticos(self, dias_criticos):
        """Gera gráficos para análise de dias críticos."""
        if len(dias_criticos) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Boxplot de margens
        dados_comparacao = [self.dados_logistica['Margem %'], dias_criticos['Margem %']]
        ax1.boxplot(dados_comparacao, 
                   labels=['Todos os Dias', 'Dias Críticos'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#3498DB', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax1.set_ylabel('Margem (%)', fontsize=12)
        ax1.set_title('Comparação de Margens', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Scatter: Custo/KM vs Margem
        if 'Custo/KM' in self.dados_logistica.columns:
            ax2.scatter(self.dados_logistica['Custo/KM'], self.dados_logistica['Margem %'], 
                      alpha=0.5, s=50, c=self.dados_logistica['Margem %'], cmap='RdYlGn')
            ax2.set_xlabel('Custo/KM (R$)', fontsize=12)
            ax2.set_ylabel('Margem (%)', fontsize=12)
            ax2.set_title('Relação: Custo/KM vs Margem', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'graficos', '05_dias_criticos.png'),  # NOME CORRIGIDO
                   dpi=250, bbox_inches='tight')
        plt.close()
    
    def _analisar_plano_acoes(self):
        """Analisa o plano de ações proposto."""
        print("   📋 Analisando plano de ações...")
        
        if self.tabela_acoes is None:
            return "❌ Tabela de ações não disponível para análise"
        
        relatorio = []
        relatorio.append("=" * 80)
        relatorio.append("PLANO DE AÇÕES")
        relatorio.append("=" * 80)
        
        relatorio.append("📋 AÇÕES PLANEJADAS:")
        relatorio.append(self.tabela_acoes.to_string(index=False))
        
        # Análise por prioridade
        relatorio.append("\n" + "=" * 80)
        relatorio.append("ANÁLISE POR PRIORIDADE")
        relatorio.append("=" * 80)
        
        if 'Prioridade' in self.tabela_acoes.columns:
            for prioridade in ['Urgente', 'Prioritária', 'Planejada']:
                acoes_prioridade = self.tabela_acoes[self.tabela_acoes['Prioridade'] == prioridade]
                if len(acoes_prioridade) > 0:
                    impacto_total = acoes_prioridade['Impacto'].sum() if 'Impacto' in acoes_prioridade.columns else 0
                    esforco_medio = acoes_prioridade['Esforço'].mean() if 'Esforço' in acoes_prioridade.columns else 0
                    prazo_medio = acoes_prioridade['Prazo'].mean() if 'Prazo' in acoes_prioridade.columns else 0
                    
                    relatorio.append(f"🎯 {prioridade.upper()}:")
                    relatorio.append(f"   Quantidade de ações: {len(acoes_prioridade)}")
                    relatorio.append(f"   Impacto total: R$ {impacto_total:,.2f}")
                    relatorio.append(f"   Esforço médio: {esforco_medio:.0f}/100")
                    relatorio.append(f"   Prazo médio: {prazo_medio:.0f} dias")
        
        # Cálculo do ROI potencial
        if 'Impacto' in self.tabela_acoes.columns and 'Margem' in self.dados_logistica.columns:
            impacto_total_acoes = self.tabela_acoes['Impacto'].sum()
            margem_atual_anual = self.dados_logistica['Margem'].sum() * (365 / len(self.dados_logistica))
            
            relatorio.append(f"\n" + "=" * 80)
            relatorio.append("IMPACTO POTENCIAL")
            relatorio.append("=" * 80)
            relatorio.append(f"💰 Margem atual (projeção anual): R$ {margem_atual_anual:,.2f}")
            relatorio.append(f"💡 Impacto potencial das ações: R$ {impacto_total_acoes:,.2f}")
            relatorio.append(f"📈 Aumento potencial: {(impacto_total_acoes/margem_atual_anual*100):.1f}%")
        
        # Gerar gráficos do plano de ações
        self._gerar_graficos_plano_acoes()
        
        return "\n".join(relatorio)
    
    def _gerar_graficos_plano_acoes(self):
        """Gera gráficos para o plano de ações."""
        if self.tabela_acoes is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico de barras - Impacto por ação
        if 'Impacto' in self.tabela_acoes.columns and 'Ação' in self.tabela_acoes.columns:
            cores_prioridade = {'Urgente': '#E74C3C', 'Prioritária': '#F39C12', 'Planejada': '#3498DB'}
            cores = [cores_prioridade.get(p, '#95A5A6') for p in self.tabela_acoes['Prioridade']] if 'Prioridade' in self.tabela_acoes.columns else '#3498DB'
            
            ax1.barh(self.tabela_acoes['Ação'], self.tabela_acoes['Impacto'], color=cores, alpha=0.8, edgecolor='black')
            ax1.set_xlabel('Impacto (R$)', fontsize=12)
            ax1.set_title('Impacto Financeiro por Ação', fontsize=14, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            
            for i, v in enumerate(self.tabela_acoes['Impacto']):
                ax1.text(v + 1000, i, f'R$ {v:,.0f}', va='center', fontweight='bold')
        
        # Scatter - Impacto vs Esforço
        if 'Impacto' in self.tabela_acoes.columns and 'Esforço' in self.tabela_acoes.columns and 'Prioridade' in self.tabela_acoes.columns:
            for prioridade in self.tabela_acoes['Prioridade'].unique():
                dados_p = self.tabela_acoes[self.tabela_acoes['Prioridade'] == prioridade]
                ax2.scatter(dados_p['Esforço'], dados_p['Impacto'], 
                          label=prioridade, s=200, alpha=0.7, 
                          color=cores_prioridade.get(prioridade, '#95A5A6'), 
                          edgecolors='black', linewidth=2)
            
            ax2.set_xlabel('Esforço', fontsize=12)
            ax2.set_ylabel('Impacto (R$)', fontsize=12)
            ax2.set_title('Matriz: Impacto vs Esforço', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'graficos', '06_plano_acoes.png'),  # NOME CORRIGIDO
                   dpi=250, bbox_inches='tight')
        plt.close()
    
    def _gerar_resumo_executivo(self):
        """Gera resumo executivo da análise."""
        print("   📄 Gerando resumo executivo...")
        
        resumo = {}
        
        # Informações básicas
        if 'Data' in self.dados_logistica.columns:
            resumo['periodo_analise'] = f"{self.dados_logistica['Data'].min().strftime('%d/%m/%Y')} a {self.dados_logistica['Data'].max().strftime('%d/%m/%Y')}"
        
        resumo['total_dias'] = len(self.dados_logistica)
        
        # Métricas principais
        if 'Margem %' in self.dados_logistica.columns:
            resumo['margem_media'] = round(self.dados_logistica['Margem %'].mean(), 2)
        
        if 'Custo/KM' in self.dados_logistica.columns:
            resumo['custo_km_medio'] = round(self.dados_logistica['Custo/KM'].mean(), 4)
        
        if 'Entregas' in self.dados_logistica.columns:
            resumo['entregas_totais'] = int(self.dados_logistica['Entregas'].sum())
        
        if 'KM Percorridos' in self.dados_logistica.columns:
            resumo['km_totais'] = int(self.dados_logistica['KM Percorridos'].sum())
        
        # Dias críticos
        if 'Margem %' in self.dados_logistica.columns:
            dias_criticos = len(self.dados_logistica[self.dados_logistica['Margem %'] < 20])
            resumo['dias_criticos'] = int(dias_criticos)
        
        # Impacto das ações
        if self.tabela_acoes is not None and 'Impacto' in self.tabela_acoes.columns:
            resumo['impacto_acoes'] = float(self.tabela_acoes['Impacto'].sum())
        
        # Salvar resumo em JSON
        with open(os.path.join(self.output_path, 'relatorios', 'resumo_executivo.json'), 'w', encoding='utf-8') as f:
            json.dump(resumo, f, indent=4, ensure_ascii=False)
        
        # Gerar relatório textual do resumo
        relatorio = []
        relatorio.append("=" * 80)
        relatorio.append("RESUMO EXECUTIVO")
        relatorio.append("=" * 80)
        
        for chave, valor in resumo.items():
            chave_formatada = chave.replace('_', ' ').title()
            relatorio.append(f"   {chave_formatada}: {valor}")
        
        return "\n".join(relatorio)
    
    def _salvar_relatorio_completo(self, resultados):
        """Salva o relatório completo em arquivo de texto."""
        print("   💾 Salvando relatório completo...")
        
        with open(os.path.join(self.output_path, 'relatorios', 'relatorio_executivo.txt'), 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE EXPLORATÓRIA - DADOS LOGÍSTICOS\n")
            f.write("=" * 80 + "\n\n")
            
            for secao, conteudo in resultados.items():
                if conteudo:
                    f.write(f"{conteudo}\n\n")
            
            # Adicionar conclusões
            f.write(self._gerar_conclusoes())
    
    def _gerar_conclusoes(self):
        """Gera conclusões e recomendações finais."""
        conclusoes = []
        conclusoes.append("=" * 80)
        conclusoes.append("CONCLUSÕES E RECOMENDAÇÕES")
        conclusoes.append("=" * 80)
        
        conclusoes.append("\n📊 PRINCIPAIS DESCOBERTAS")
        conclusoes.append("\n1. **Rentabilidade**")
        conclusoes.append("   - Análise completa das margens e identificação de dias críticos")
        conclusoes.append("   - Tendências temporais e sazonalidades identificadas")
        conclusoes.append("   - Correlações entre variáveis operacionais mapeadas")
        
        conclusoes.append("\n2. **Custos**")
        conclusoes.append("   - Composição detalhada dos custos operacionais")
        conclusoes.append("   - Identificação dos principais drivers de custo")
        conclusoes.append("   - Análise de eficiência por quilômetro")
        
        conclusoes.append("\n3. **Eficiência Operacional**")
        conclusoes.append("   - Métricas de entregas e quilometragem analisadas")
        conclusoes.append("   - Relação entre volume operacional e rentabilidade")
        conclusoes.append("   - Identificação de oportunidades de otimização")
        
        if self.tabela_acoes is not None:
            conclusoes.append("\n🎯 RECOMENDAÇÕES PRIORITÁRIAS")
            conclusoes.append("\n**Ações Urgentes (30 dias)**")
            conclusoes.append("1. Implementar as ações de alto impacto identificadas")
            conclusoes.append("2. Monitorar dias críticos com atenção")
            conclusoes.append("3. Otimizar rotas para reduzir custo por KM")
            
            conclusoes.append("\n**Ações de Médio Prazo (60-90 dias)**")
            conclusoes.append("4. Revisar composição de custos")
            conclusoes.append("5. Implementar sistema de alertas para margens baixas")
            
            conclusoes.append("\n**Ações de Longo Prazo (180 dias)**")
            conclusoes.append("6. Desenvolver modelo preditivo para rentabilidade")
            conclusoes.append("7. Implementar dashboard em tempo real")
        
        conclusoes.append("\n💡 PRÓXIMOS PASSOS")
        conclusoes.append("1. Realizar análise preditiva para prever rentabilidade")
        conclusoes.append("2. Implementar sistema de monitoramento contínuo")
        conclusoes.append("3. Validar ações propostas com testes piloto")
        conclusoes.append("4. Atualizar análise mensalmente")
        
        return "\n".join(conclusoes)


# Função auxiliar para uso direto
def executar_analise_exploratoria(dados_logistica, dados_mensais=None, tabela_acoes=None, output_path='output'):
    """
    Função conveniente para executar análise exploratória.
    
    Args:
        dados_logistica (pd.DataFrame): Dados diários de logística
        dados_mensais (pd.DataFrame): Dados mensais para referência
        tabela_acoes (pd.DataFrame): Tabela de ações propostas
        output_path (str): Pasta de saída
        
    Returns:
        dict: Resultados da análise
    """
    analise = AnaliseExploratoria(dados_logistica, dados_mensais, tabela_acoes, output_path)
    return analise.gerar_relatorio_completo()


if __name__ == "__main__":
    # Exemplo de uso direto
    print("🔧 Testando módulo de análise exploratória...")
    
    # Carregar dados de exemplo (substitua por seus dados)
    try:
        dados_logistica = pd.read_csv('data/dados_logistica.csv')
        dados_logistica['Data'] = pd.to_datetime(dados_logistica['Data'])
        
        dados_mensais = pd.read_csv('data/dados_mensais.csv') if os.path.exists('data/dados_mensais.csv') else None
        tabela_acoes = pd.read_csv('data/tabela_acoes.csv') if os.path.exists('data/tabela_acoes.csv') else None
        
        # Executar análise
        resultados = executar_analise_exploratoria(dados_logistica, dados_mensais, tabela_acoes)
        print("✅ Teste concluído com sucesso!")
        
    except FileNotFoundError:
        print("❌ Arquivos de dados não encontrados. Execute a partir do main.py.")