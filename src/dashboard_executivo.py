"""
Módulo para geração de dashboard executivo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


class DashboardExecutivo:
    """Classe para geração de dashboard executivo."""
    
    def __init__(self, dados_logistica, dados_mensais, tabela_acoes, output_path='output'):
        """
        Inicializa o dashboard executivo.
        
        Args:
            dados_logistica (DataFrame): Dados diários de logística
            dados_mensais (DataFrame): Dados mensais agregados
            tabela_acoes (DataFrame): Tabela de ações propostas
            output_path (str): Caminho para salvar os outputs
        """
        self.dados_logistica = dados_logistica
        self.dados_mensais = dados_mensais
        self.tabela_acoes = tabela_acoes
        self.output_path = output_path
        self.graficos_path = os.path.join(output_path, 'graficos')
        self.relatorios_path = os.path.join(output_path, 'relatorios')
        
        # Criar pastas se não existirem
        os.makedirs(self.graficos_path, exist_ok=True)
        os.makedirs(self.relatorios_path, exist_ok=True)
        
        # Configurar estilo
        sns.set_style('whitegrid')
    
    def calcular_kpis(self):
        """
        Calcula os principais KPIs do negócio.
        
        Returns:
            dict: Dicionário com os KPIs
        """
        print("=" * 80)
        print("CÁLCULO DOS KPIs")
        print("=" * 80)
        
        kpis = {
            # Receitas e Custos
            'frete_total': self.dados_logistica['Frete'].sum(),
            'custo_total': self.dados_logistica['Custo Total'].sum(),
            'margem_total': self.dados_logistica['Margem'].sum(),
            'margem_percentual_media': self.dados_logistica['Margem %'].mean(),
            
            # Operacionais
            'km_total': self.dados_logistica['KM Percorridos'].sum(),
            'entregas_total': self.dados_logistica['Entregas'].sum(),
            'peso_total': self.dados_logistica['Peso (ton)'].sum(),
            
            # Médias
            'frete_medio_dia': self.dados_logistica['Frete'].mean(),
            'custo_medio_dia': self.dados_logistica['Custo Total'].mean(),
            'margem_media_dia': self.dados_logistica['Margem'].mean(),
            'entregas_media_dia': self.dados_logistica['Entregas'].mean(),
            
            # Eficiência
            'custo_por_km': self.dados_logistica['Custo/KM'].mean(),
            'km_por_entrega': self.dados_logistica['KM/Entrega'].mean(),
            'frete_por_km': (self.dados_logistica['Frete'].sum() / 
                           self.dados_logistica['KM Percorridos'].sum()),
            
            # Composição de Custos
            'custo_combustivel_total': self.dados_logistica['Custo Combustível'].sum(),
            'custo_manutencao_total': self.dados_logistica['Custo Manutenção'].sum(),
            'custo_motorista_total': self.dados_logistica['Custo Motorista'].sum(),
        }
        
        # Calcular percentuais de composição
        kpis['perc_combustivel'] = (kpis['custo_combustivel_total'] / kpis['custo_total']) * 100
        kpis['perc_manutencao'] = (kpis['custo_manutencao_total'] / kpis['custo_total']) * 100
        kpis['perc_motorista'] = (kpis['custo_motorista_total'] / kpis['custo_total']) * 100
        
        # Exibir KPIs
        print("\n💰 FINANCEIRO:")
        print(f"   Frete Total: R$ {kpis['frete_total']:,.2f}")
        print(f"   Custo Total: R$ {kpis['custo_total']:,.2f}")
        print(f"   Margem Total: R$ {kpis['margem_total']:,.2f}")
        print(f"   Margem %: {kpis['margem_percentual_media']:.2f}%")
        
        print("\n📊 OPERACIONAL:")
        print(f"   KM Total: {kpis['km_total']:,.0f} km")
        print(f"   Entregas Total: {kpis['entregas_total']:,.0f}")
        print(f"   Peso Total: {kpis['peso_total']:,.2f} ton")
        
        print("\n📈 EFICIÊNCIA:")
        print(f"   Custo por KM: R$ {kpis['custo_por_km']:.2f}")
        print(f"   KM por Entrega: {kpis['km_por_entrega']:.2f} km")
        print(f"   Frete por KM: R$ {kpis['frete_por_km']:.2f}")
        
        return kpis
    
    def gerar_dashboard_principal(self):
        """Gera o dashboard principal com os KPIs."""
        print("\n" + "=" * 80)
        print("GERANDO DASHBOARD PRINCIPAL")
        print("=" * 80)
        
        kpis = self.calcular_kpis()
        
        # Criar figura com múltiplos subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. KPIs Principais (texto)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        kpi_text = f"""
        DASHBOARD EXECUTIVO - ANÁLISE LOGÍSTICA
        Período: {self.dados_logistica['Data'].min().strftime('%d/%m/%Y')} a {self.dados_logistica['Data'].max().strftime('%d/%m/%Y')}
        
        💰 FINANCEIRO                           📊 OPERACIONAL                          📈 EFICIÊNCIA
        Frete Total: R$ {kpis['frete_total']:,.2f}        KM Total: {kpis['km_total']:,.0f} km              Custo/KM: R$ {kpis['custo_por_km']:.2f}
        Custo Total: R$ {kpis['custo_total']:,.2f}        Entregas: {kpis['entregas_total']:,.0f}                  KM/Entrega: {kpis['km_por_entrega']:.2f} km
        Margem: R$ {kpis['margem_total']:,.2f}             Peso: {kpis['peso_total']:,.2f} ton                Frete/KM: R$ {kpis['frete_por_km']:.2f}
        Margem %: {kpis['margem_percentual_media']:.2f}%
        """
        
        ax1.text(0.5, 0.5, kpi_text, ha='center', va='center', 
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 2. Evolução da Margem %
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.dados_mensais['Mês'], self.dados_mensais['Margem %'], 
                marker='o', linewidth=2, markersize=8, color='#2ecc71')
        ax2.axhline(y=kpis['margem_percentual_media'], color='r', 
                   linestyle='--', label=f'Média: {kpis["margem_percentual_media"]:.1f}%')
        ax2.set_title('Evolução da Margem %', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Mês')
        ax2.set_ylabel('Margem %')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Composição de Custos
        ax3 = fig.add_subplot(gs[1, 1])
        custos = [kpis['custo_combustivel_total'], 
                 kpis['custo_manutencao_total'], 
                 kpis['custo_motorista_total']]
        labels = ['Combustível\n48.8%', 'Manutenção\n19.8%', 'Motorista\n31.4%']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax3.pie(custos, labels=labels, colors=colors, autopct='R$ %.0f',
               startangle=90, textprops={'fontsize': 9})
        ax3.set_title('Composição de Custos', fontsize=12, fontweight='bold')
        
        # 4. Frete vs Custo Mensal
        ax4 = fig.add_subplot(gs[1, 2])
        x = np.arange(len(self.dados_mensais))
        width = 0.35
        ax4.bar(x - width/2, self.dados_mensais['Frete'], width, 
               label='Frete', color='#3498db', alpha=0.8)
        ax4.bar(x + width/2, self.dados_mensais['Custo Total'], width, 
               label='Custo', color='#e74c3c', alpha=0.8)
        ax4.set_title('Frete vs Custo Mensal', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Mês')
        ax4.set_ylabel('Valor (R$)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.dados_mensais['Mês'], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. KM Percorridos Mensal
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.bar(self.dados_mensais['Mês'], self.dados_mensais['KM Percorridos'], 
               color='#9b59b6', alpha=0.7)
        ax5.set_title('KM Percorridos por Mês', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Mês')
        ax5.set_ylabel('KM')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Entregas Mensais
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.bar(self.dados_mensais['Mês'], self.dados_mensais['Entregas'], 
               color='#f39c12', alpha=0.7)
        ax6.set_title('Entregas por Mês', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Mês')
        ax6.set_ylabel('Número de Entregas')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Custo/KM Mensal
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.plot(self.dados_mensais['Mês'], self.dados_mensais['Custo/KM'], 
                marker='s', linewidth=2, markersize=8, color='#e74c3c')
        ax7.set_title('Custo por KM Mensal', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Mês')
        ax7.set_ylabel('Custo/KM (R$)')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('DASHBOARD EXECUTIVO - LOGÍSTICA', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(os.path.join(self.graficos_path, '07_dashboard_executivo.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dashboard salvo: {os.path.join(self.graficos_path, '07_dashboard_executivo.png')}")
        
        return kpis
    
    def analisar_acoes_propostas(self):
        """Analisa e visualiza as ações propostas."""
        print("\n" + "=" * 80)
        print("ANÁLISE DAS AÇÕES PROPOSTAS")
        print("=" * 80)
        
        # Ordenar por impacto
        acoes_ordenadas = self.tabela_acoes.sort_values('Impacto', ascending=False)
        
        print("\n📋 Top 5 Ações por Impacto:")
        for idx, row in acoes_ordenadas.head(5).iterrows():
            print(f"   {row['Ação']}: R$ {row['Impacto']:,.2f} ({row['Prioridade']})")
        
        # Gráfico de ações
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Gráfico 1: Impacto das ações
        top_acoes = acoes_ordenadas.head(10)
        axes[0].barh(top_acoes['Ação'], top_acoes['Impacto'], color='#3498db', alpha=0.7)
        axes[0].set_xlabel('Impacto (R$)')
        axes[0].set_title('Top 10 Ações por Impacto Financeiro', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Gráfico 2: Impacto vs Esforço
        cores_prioridade = {
            'Urgente': '#e74c3c',
            'Prioritária': '#f39c12',
            'Planejada': '#3498db'
        }
        
        for prioridade in self.tabela_acoes['Prioridade'].unique():
            dados_prioridade = self.tabela_acoes[self.tabela_acoes['Prioridade'] == prioridade]
            axes[1].scatter(dados_prioridade['Esforço'], dados_prioridade['Impacto'],
                          s=200, alpha=0.6, label=prioridade,
                          color=cores_prioridade.get(prioridade, '#95a5a6'))
        
        axes[1].set_xlabel('Esforço (%)')
        axes[1].set_ylabel('Impacto (R$)')
        axes[1].set_title('Matriz Impacto vs Esforço', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graficos_path, '08_acoes_propostas.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Gráfico salvo: {os.path.join(self.graficos_path, '08_acoes_propostas.png')}")
        
        # Calcular impacto total
        impacto_total = self.tabela_acoes['Impacto'].sum()
        impacto_urgente = self.tabela_acoes[
            self.tabela_acoes['Prioridade'] == 'Urgente'
        ]['Impacto'].sum()
        
        print(f"\n💰 Impacto Total das Ações: R$ {impacto_total:,.2f}")
        print(f"🔴 Impacto das Ações Urgentes: R$ {impacto_urgente:,.2f}")
        
        return acoes_ordenadas
    
    def gerar_relatorio_executivo(self):
        """Gera relatório executivo completo em texto."""
        print("\n" + "=" * 80)
        print("GERANDO RELATÓRIO EXECUTIVO")
        print("=" * 80)
        
        kpis = self.calcular_kpis()
        
        relatorio = f"""
{'='*80}
RELATÓRIO EXECUTIVO - ANÁLISE LOGÍSTICA
{'='*80}

Período de Análise: {self.dados_logistica['Data'].min().strftime('%d/%m/%Y')} a {self.dados_logistica['Data'].max().strftime('%d/%m/%Y')}
Data do Relatório: {datetime.now().strftime('%d/%m/%Y %H:%M')}

{'='*80}
1. RESUMO EXECUTIVO
{'='*80}

A análise dos dados logísticos revela uma operação com margem média de {kpis['margem_percentual_media']:.2f}%,
gerando uma margem total de R$ {kpis['margem_total']:,.2f} no período analisado.

Principais Indicadores:
- Frete Total: R$ {kpis['frete_total']:,.2f}
- Custo Total: R$ {kpis['custo_total']:,.2f}
- Margem Total: R$ {kpis['margem_total']:,.2f}
- Margem Percentual Média: {kpis['margem_percentual_media']:.2f}%

{'='*80}
2. ANÁLISE OPERACIONAL
{'='*80}

Volume de Operações:
- Total de KM Percorridos: {kpis['km_total']:,.0f} km
- Total de Entregas: {kpis['entregas_total']:,.0f}
- Peso Total Transportado: {kpis['peso_total']:,.2f} toneladas

Médias Diárias:
- Entregas por Dia: {kpis['entregas_media_dia']:.0f}
- Frete por Dia: R$ {kpis['frete_medio_dia']:,.2f}
- Custo por Dia: R$ {kpis['custo_medio_dia']:,.2f}

{'='*80}
3. ANÁLISE DE CUSTOS
{'='*80}

Composição dos Custos:
- Combustível: R$ {kpis['custo_combustivel_total']:,.2f} ({kpis['perc_combustivel']:.1f}%)
- Manutenção: R$ {kpis['custo_manutencao_total']:,.2f} ({kpis['perc_manutencao']:.1f}%)
- Motorista: R$ {kpis['custo_motorista_total']:,.2f} ({kpis['perc_motorista']:.1f}%)

O combustível representa o maior componente de custo ({kpis['perc_combustivel']:.1f}%),
seguido pelos custos com motoristas ({kpis['perc_motorista']:.1f}%) e manutenção ({kpis['perc_manutencao']:.1f}%).

{'='*80}
4. INDICADORES DE EFICIÊNCIA
{'='*80}

- Custo por KM: R$ {kpis['custo_por_km']:.2f}
- Frete por KM: R$ {kpis['frete_por_km']:.2f}
- KM por Entrega: {kpis['km_por_entrega']:.2f} km

{'='*80}
5. RECOMENDAÇÕES ESTRATÉGICAS
{'='*80}

Com base na análise realizada, recomenda-se:

1. OTIMIZAÇÃO DE CUSTOS DE COMBUSTÍVEL
   - Implementar sistema de monitoramento de consumo
   - Treinar motoristas em direção econômica
   - Avaliar rotas alternativas mais eficientes

2. MELHORIA DA MARGEM OPERACIONAL
   - Renegociar contratos com margens abaixo da média
   - Implementar precificação dinâmica baseada em distância e peso
   - Reduzir dias com margem negativa

3. EFICIÊNCIA OPERACIONAL
   - Otimizar rotas para reduzir KM/Entrega
   - Implementar manutenção preventiva para reduzir custos
   - Aumentar taxa de ocupação dos veículos

4. GESTÃO DE PERFORMANCE
   - Estabelecer metas de margem mínima por operação
   - Monitorar KPIs diariamente
   - Implementar sistema de alertas para anomalias

{'='*80}
6. PRÓXIMOS PASSOS
{'='*80}

- Implementar as ações urgentes identificadas
- Estabelecer sistema de monitoramento contínuo
- Revisar precificação de serviços
- Investir em tecnologia para otimização de rotas

{'='*80}
FIM DO RELATÓRIO
{'='*80}
"""
        
        # Salvar relatório
        caminho_relatorio = os.path.join(self.relatorios_path, 'relatorio_executivo.txt')
        with open(caminho_relatorio, 'w', encoding='utf-8') as f:
            f.write(relatorio)
        
        print(relatorio)
        print(f"\n✅ Relatório salvo: {caminho_relatorio}")
        
        return relatorio
    
    def gerar_dashboard_completo(self):
        """Gera dashboard executivo completo."""
        print("\n" + "=" * 80)
        print("GERANDO DASHBOARD EXECUTIVO COMPLETO")
        print("=" * 80)
        
        # Gerar todos os componentes
        kpis = self.gerar_dashboard_principal()
        acoes = self.analisar_acoes_propostas()
        relatorio = self.gerar_relatorio_executivo()
        
        print("\n" + "=" * 80)
        print("✅ DASHBOARD EXECUTIVO CONCLUÍDO!")
        print("=" * 80)
        print(f"📁 Gráficos salvos em: {self.graficos_path}")
        print(f"📁 Relatórios salvos em: {self.relatorios_path}")
        
        return {
            'kpis': kpis,
            'acoes': acoes,
            'relatorio': relatorio
        }
