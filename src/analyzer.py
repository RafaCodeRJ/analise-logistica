#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Análise de Dados
===========================

Contém funções para análise estatística e cálculo de KPIs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogisticsAnalyzer:
    """Classe para análise de dados logísticos."""
    
    def __init__(self, dados_logistica: pd.DataFrame):
        """
        Inicializa o analisador.
        
        Args:
            dados_logistica: DataFrame com dados logísticos
        """
        self.dados = dados_logistica.copy()
        self._prepare_data()
    
    def _prepare_data(self) -> None:
        """Prepara os dados para análise."""
        if 'Data' in self.dados.columns:
            self.dados['Data'] = pd.to_datetime(self.dados['Data'])
            self.dados = self.dados.sort_values('Data')
        logger.info("✅ Dados preparados para análise")
    
    def calculate_kpis(self) -> Dict[str, float]:
        """
        Calcula os principais KPIs.
        
        Returns:
            Dicionário com os KPIs calculados
        """
        kpis = {
            'margem_media': self.dados['Margem %'].mean(),
            'margem_mediana': self.dados['Margem %'].median(),
            'margem_min': self.dados['Margem %'].min(),
            'margem_max': self.dados['Margem %'].max(),
            'custo_km_medio': self.dados['Custo/KM'].mean(),
            'custo_total': self.dados['Custo Total'].sum(),
            'frete_total': self.dados['Frete'].sum(),
            'margem_total': self.dados['Margem'].sum(),
            'entregas_totais': self.dados['Entregas'].sum(),
            'km_totais': self.dados['KM Percorridos'].sum(),
            'peso_total': self.dados['Peso (ton)'].sum(),
            'dias_criticos': len(self.dados[self.dados['Margem %'] < 20]),
            'percentual_dias_criticos': len(self.dados[self.dados['Margem %'] < 20]) / len(self.dados) * 100
        }
        
        logger.info("✅ KPIs calculados")
        return kpis
    
    def analyze_costs(self) -> Dict[str, any]:
        """
        Analisa a composição de custos.
        
        Returns:
            Dicionário com análise de custos
        """
        custo_combustivel = self.dados['Custo Combustível'].sum()
        custo_manutencao = self.dados['Custo Manutenção'].sum()
        custo_motorista = self.dados['Custo Motorista'].sum()
        custo_total = self.dados['Custo Total'].sum()
        
        analise = {
            'custo_combustivel': custo_combustivel,
            'custo_manutencao': custo_manutencao,
            'custo_motorista': custo_motorista,
            'custo_total': custo_total,
            'prop_combustivel': custo_combustivel / custo_total * 100,
            'prop_manutencao': custo_manutencao / custo_total * 100,
            'prop_motorista': custo_motorista / custo_total * 100,
            'custo_medio_dia': self.dados['Custo Total'].mean(),
            'custo_mediano_dia': self.dados['Custo Total'].median()
        }
        
        logger.info("✅ Análise de custos concluída")
        return analise
    
    def classify_days(self) -> Dict[str, int]:
        """
        Classifica os dias por rentabilidade.
        
        Returns:
            Dicionário com contagem de dias por classificação
        """
        classificacao = {
            'excelente': len(self.dados[self.dados['Margem %'] >= 35]),
            'bom': len(self.dados[(self.dados['Margem %'] >= 25) & (self.dados['Margem %'] < 35)]),
            'regular': len(self.dados[(self.dados['Margem %'] >= 20) & (self.dados['Margem %'] < 25)]),
            'critico': len(self.dados[self.dados['Margem %'] < 20])
        }
        
        logger.info("✅ Classificação de dias concluída")
        return classificacao
    
    def identify_critical_days(self, threshold: float = 20.0) -> pd.DataFrame:
        """
        Identifica dias críticos.
        
        Args:
            threshold: Limite de margem para considerar crítico
            
        Returns:
            DataFrame com dias críticos
        """
        dias_criticos = self.dados[self.dados['Margem %'] < threshold].copy()
        dias_criticos = dias_criticos.sort_values('Margem %')
        
        logger.info(f"✅ Identificados {len(dias_criticos)} dias críticos")
        return dias_criticos
    
    def calculate_trends(self, window: int = 30) -> pd.DataFrame:
        """
        Calcula tendências usando médias móveis.
        
        Args:
            window: Janela para média móvel
            
        Returns:
            DataFrame com tendências
        """
        trends = pd.DataFrame()
        trends['Data'] = self.dados['Data']
        trends['Margem_MA'] = self.dados['Margem %'].rolling(window=window).mean()
        trends['Custo_KM_MA'] = self.dados['Custo/KM'].rolling(window=window).mean()
        trends['Entregas_MA'] = self.dados['Entregas'].rolling(window=window).mean()
        
        logger.info(f"✅ Tendências calculadas (janela={window})")
        return trends
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Retorna estatísticas descritivas.
        
        Returns:
            DataFrame com estatísticas
        """
        colunas_numericas = self.dados.select_dtypes(include=[np.number]).columns
        stats = self.dados[colunas_numericas].describe()
        
        logger.info("✅ Estatísticas descritivas calculadas")
        return stats


class ActionAnalyzer:
    """Classe para análise do plano de ações."""
    
    def __init__(self, tabela_acoes: pd.DataFrame):
        """
        Inicializa o analisador de ações.
        
        Args:
            tabela_acoes: DataFrame com plano de ações
        """
        self.acoes = tabela_acoes.copy()
    
    def analyze_by_priority(self) -> Dict[str, Dict]:
        """
        Analisa ações por prioridade.
        
        Returns:
            Dicionário com análise por prioridade
        """
        analise = {}
        
        for prioridade in self.acoes['Prioridade'].unique():
            acoes_p = self.acoes[self.acoes['Prioridade'] == prioridade]
            analise[prioridade] = {
                'quantidade': len(acoes_p),
                'impacto_total': acoes_p['Impacto'].sum(),
                'esforco_medio': acoes_p['Esforço'].mean(),
                'prazo_medio': acoes_p['Prazo'].mean()
            }
        
        logger.info("✅ Análise por prioridade concluída")
        return analise
    
    def calculate_roi(self) -> Dict[str, float]:
        """
        Calcula métricas de ROI.
        
        Returns:
            Dicionário com métricas de ROI
        """
        roi = {
            'impacto_total': self.acoes['Impacto'].sum(),
            'impacto_urgente': self.acoes[self.acoes['Prioridade'] == 'Urgente']['Impacto'].sum(),
            'impacto_prioritario': self.acoes[self.acoes['Prioridade'] == 'Prioritária']['Impacto'].sum(),
            'esforco_medio': self.acoes['Esforço'].mean(),
            'prazo_medio': self.acoes['Prazo'].mean()
        }
        
        logger.info("✅ ROI calculado")
        return roi


if __name__ == '__main__':
    # Teste do módulo
    from data_loader import load_data
    
    try:
        dados_log, dados_men, tab_acoes = load_data()
        
        # Testar LogisticsAnalyzer
        analyzer = LogisticsAnalyzer(dados_log)
        kpis = analyzer.calculate_kpis()
        print("\n📊 KPIs:", kpis)
        
        # Testar ActionAnalyzer
        action_analyzer = ActionAnalyzer(tab_acoes)
        roi = action_analyzer.calculate_roi()
        print("\n💰 ROI:", roi)
        
        print("\n✅ Teste do módulo concluído com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
