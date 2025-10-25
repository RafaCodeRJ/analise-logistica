"""
Script principal para executar todas as análises do projeto de logística.

Este script executa:

1. Carregamento dos dados
2. Análise Exploratória
3. Análise Preditiva
4. Dashboard Executivo
"""

from src.dashboard_executivo import DashboardExecutivo
from src.analise_preditiva import AnalisePreditiva
from src.analise_exploratoria import AnaliseExploratoria
from src.data_loader import DataLoader
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Adicionar pasta src ao path
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), 'src'))


def print_header(titulo):
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 80)
    print(f"  {titulo}")
    print("=" * 80 + "\n")


def main():
    """Função principal que executa todas as análises."""

    print_header("🚀 INICIANDO ANÁLISE COMPLETA DE DADOS LOGÍSTICOS")

    try:
        # ========================================================================
        # 1. CARREGAMENTO DOS DADOS
        # ========================================================================
        print_header("📂 ETAPA 1: CARREGAMENTO DOS DADOS")

        loader = DataLoader(data_path='data')
        dados_logistica, dados_mensais, tabela_acoes = loader.carregar_dados()

        # Exibir estatísticas básicas
        stats = loader.get_estatisticas_basicas()
        print(f"\n📊 Estatísticas Básicas:")
        print(f"   Total de registros: {stats['total_registros']}")
        print(f"   Margem média: {stats['margem_media']:.2f}%")
        print(f"   Custo total médio: R$ {stats['custo_total_medio']:,.2f}")
        print(f"   Frete médio: R$ {stats['frete_medio']:,.2f}")

        # ========================================================================
        # 2. ANÁLISE EXPLORATÓRIA
        # ========================================================================
        print_header("🔍 ETAPA 2: ANÁLISE EXPLORATÓRIA")

        analise_exp = AnaliseExploratoria(
            dados_logistica=dados_logistica,
            dados_mensais=dados_mensais,
            output_path='output'
        )

        resultados_exp = analise_exp.gerar_relatorio_completo()

        # ========================================================================
        # 3. ANÁLISE PREDITIVA
        # ========================================================================
        print_header("🤖 ETAPA 3: ANÁLISE PREDITIVA E MODELAGEM")

        analise_pred = AnalisePreditiva(
            dados_logistica=dados_logistica,
            output_path='output'
        )

        resultados_pred = analise_pred.gerar_relatorio_completo()

        # ========================================================================
        # 4. DASHBOARD EXECUTIVO
        # ========================================================================
        print_header("📊 ETAPA 4: DASHBOARD EXECUTIVO")

        dashboard = DashboardExecutivo(
            dados_logistica=dados_logistica,
            dados_mensais=dados_mensais,
            tabela_acoes=tabela_acoes,
            output_path='output'
        )

        resultados_dash = dashboard.gerar_dashboard_completo()

        # ========================================================================
        # RESUMO FINAL
        # ========================================================================
        print_header("✅ ANÁLISE COMPLETA FINALIZADA COM SUCESSO!")

        print("📁 ARQUIVOS GERADOS:")
        print("\n   Gráficos:")
        print("   - output/graficos/01_composicao_custos.png")
        print("   - output/graficos/02_evolucao_temporal.png")
        print("   - output/graficos/03_matriz_correlacao.png")
        print("   - output/graficos/04_analise_dispersao.png")
        print("   - output/graficos/05_importancia_features.png")
        print("   - output/graficos/06_previsoes_modelo.png")
        print("   - output/graficos/07_dashboard_executivo.png")
        print("   - output/graficos/08_acoes_propostas.png")

        print("\n   Modelos:")
        print("   - output/modelos/melhor_modelo.pkl")

        print("\n   Relatórios:")
        print("   - output/relatorios/relatorio_executivo.txt")

        print("\n" + "=" * 80)
        print("📊 PRINCIPAIS RESULTADOS:")
        print("=" * 80)

        kpis = resultados_dash['kpis']
        print(f"\n💰 FINANCEIRO:")
        print(f"   Margem Média: {kpis['margem_percentual_media']:.2f}%")
        print(f"   Margem Total: R$ {kpis['margem_total']:,.2f}")
        print(f"   Frete Total: R$ {kpis['frete_total']:,.2f}")
        print(f"   Custo Total: R$ {kpis['custo_total']:,.2f}")

        print(f"\n📊 OPERACIONAL:")
        print(f"   KM Total: {kpis['km_total']:,.0f} km")
        print(f"   Entregas Total: {kpis['entregas_total']:,.0f}")
        print(f"   Peso Total: {kpis['peso_total']:,.2f} ton")

        print(f"\n📈 EFICIÊNCIA:")
        print(f"   Custo por KM: R$ {kpis['custo_por_km']:.2f}")
        print(f"   KM por Entrega: {kpis['km_por_entrega']:.2f} km")
        print(f"   Frete por KM: R$ {kpis['frete_por_km']:.2f}")

        metricas = resultados_pred['metricas']
        melhor_modelo = metricas.loc[metricas['R²_Test'].idxmax()]
        print(f"\n🤖 MODELO PREDITIVO:")
        print(f"   Melhor Modelo: {melhor_modelo['Model']}")
        print(f"   R² (Teste): {melhor_modelo['R²_Test']:.4f}")
        print(f"   RMSE (Teste): R$ {melhor_modelo['RMSE_Test']:.2f}")

        print("\n" + "=" * 80)
        print("🎉 PROJETO CONCLUÍDO COM SUCESSO!")
        print("=" * 80)

        return {
            'dados': (dados_logistica, dados_mensais, tabela_acoes),
            'analise_exploratoria': resultados_exp,
            'analise_preditiva': resultados_pred,
            'dashboard': resultados_dash
        }

    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    resultados = main()

    if resultados is not None:
        print("\n✅ Todas as análises foram executadas com sucesso!")
        print("📁 Verifique a pasta 'output' para os resultados.")
    else:
        print("\n❌ Houve um erro na execução. Verifique os logs acima.")
