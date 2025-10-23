#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Visualização
======================

Contém funções para criar gráficos e visualizações.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações globais
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


class LogisticsVisualizer:
    """Classe para visualização de dados logísticos."""
    
    def __init__(self, dados_logistica: pd.DataFrame):
        """
        Inicializa o visualizador.
        
        Args:
            dados_logistica: DataFrame com dados logísticos
        """
        self.dados = dados_logistica.copy()
        if 'Data' in self.dados.columns:
            self.dados['Data'] = pd.to_datetime(self.dados['Data'])
            self.dados = self.dados.sort_values('Data')
    
    def plot_margin_evolution(self, figsize: Tuple[int, int] = (14, 6)) -> None:
        """
        Plota evolução da margem ao longo do tempo.
        
        Args:
            figsize: Tamanho da figura
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.dados['Data'], self.dados['Margem %'], 
                color='#2ECC71', linewidth=2, alpha=0.7)
        ax.fill_between(self.dados['Data'], self.dados['Margem %'], 
                        alpha=0.3, color='#2ECC71')
        ax.axhline(y=self.dados['Margem %'].mean(), color='red', 
                   linestyle='--', linewidth=2, 
                   label=f'Média: {self.dados["Margem %"].mean():.1f}%')
        ax.axhline(y=20, color='orange', linestyle='--', linewidth=2, 
                   label='Limite Crítico: 20%')
        
        ax.set_xlabel('Data', fontsize=12, fontweight='bold')
        ax.set_ylabel('Margem (%)', fontsize=12, fontweight='bold')
        ax.set_title('Evolução da Margem de Lucro', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        logger.info("✅ Gráfico de evolução da margem criado")
    
    def plot_cost_composition(self, figsize: Tuple[int, int] = (14, 6)) -> None:
        """
        Plota composição de custos.
        
        Args:
            figsize: Tamanho da figura
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Pizza
        custos = [
            self.dados['Custo Combustível'].sum(),
            self.dados['Custo Manutenção'].sum(),
            self.dados['Custo Motorista'].sum()
        ]
        labels = ['Combustível', 'Manutenção', 'Motorista']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        ax1.pie(custos, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90, shadow=True)
        ax1.set_title('Composição dos Custos', fontsize=14, fontweight='bold')
        
        # Barras
        custos_medios = {
            'Combustível': self.dados['Custo Combustível'].mean(),
            'Manutenção': self.dados['Custo Manutenção'].mean(),
            'Motorista': self.dados['Custo Motorista'].mean()
        }
        
        ax2.bar(custos_medios.keys(), custos_medios.values(), 
                color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Valor Médio (R$)', fontsize=12)
        ax2.set_title('Custos Médios Diários', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for i, (k, v) in enumerate(custos_medios.items()):
            ax2.text(i, v + 50, f'R$ {v:,.0f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        logger.info("✅ Gráfico de composição de custos criado")
    
    def plot_margin_distribution(self, figsize: Tuple[int, int] = (14, 6)) -> None:
        """
        Plota distribuição das margens.
        
        Args:
            figsize: Tamanho da figura
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histograma
        ax1.hist(self.dados['Margem %'], bins=30, color='#45B7D1', 
                alpha=0.7, edgecolor='black')
        ax1.axvline(self.dados['Margem %'].mean(), color='red', 
                   linestyle='--', linewidth=2, 
                   label=f'Média: {self.dados["Margem %"].mean():.1f}%')
        ax1.axvline(self.dados['Margem %'].median(), color='green', 
                   linestyle='--', linewidth=2, 
                   label=f'Mediana: {self.dados["Margem %"].median():.1f}%')
        ax1.set_xlabel('Margem (%)', fontsize=12)
        ax1.set_ylabel('Frequência', fontsize=12)
        ax1.set_title('Distribuição das Margens', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Boxplot
        ax2.boxplot(self.dados['Margem %'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#45B7D1', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Margem (%)', fontsize=12)
        ax2.set_title('Boxplot das Margens', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        logger.info("✅ Gráfico de distribuição de margens criado")
    
    def plot_correlation_matrix(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plota matriz de correlação.
        
        Args:
            figsize: Tamanho da figura
        """
        colunas = ['Entregas', 'KM Percorridos', 'Peso (ton)', 
                   'Custo Total', 'Frete', 'Margem %']
        correlacao = self.dados[colunas].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(correlacao, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8})
        plt.title('Matriz de Correlação', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        logger.info("✅ Matriz de correlação criada")
    
    def plot_dashboard(self, figsize: Tuple[int, int] = (18, 10)) -> None:
        """
        Cria dashboard completo.
        
        Args:
            figsize: Tamanho da figura
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Evolução da margem
        axes[0, 0].plot(self.dados['Data'], self.dados['Margem %'], 
                       color='#2ECC71', linewidth=2)
        axes[0, 0].axhline(y=self.dados['Margem %'].mean(), 
                          color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Evolução da Margem', fontweight='bold')
        axes[0, 0].set_ylabel('Margem (%)')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Custos vs Receitas
        axes[0, 1].plot(self.dados['Data'], self.dados['Custo Total'], 
                       label='Custo', color='#E74C3C', linewidth=2)
        axes[0, 1].plot(self.dados['Data'], self.dados['Frete'], 
                       label='Frete', color='#3498DB', linewidth=2)
        axes[0, 1].set_title('Custos vs Receitas', fontweight='bold')
        axes[0, 1].set_ylabel('Valor (R$)')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Distribuição de margens
        axes[1, 0].hist(self.dados['Margem %'], bins=25, 
                       color='#9B59B6', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(self.dados['Margem %'].mean(), 
                          color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Distribuição de Margens', fontweight='bold')
        axes[1, 0].set_xlabel('Margem (%)')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Custo/KM
        axes[1, 1].plot(self.dados['Data'], self.dados['Custo/KM'], 
                       color='#F39C12', linewidth=2)
        axes[1, 1].axhline(y=self.dados['Custo/KM'].mean(), 
                          color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Custo por KM', fontweight='bold')
        axes[1, 1].set_ylabel('Custo/KM (R$)')
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle('DASHBOARD LOGÍSTICO', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
        logger.info("✅ Dashboard criado")


if __name__ == '__main__':
    # Teste do módulo
    from data_loader import load_data
    
    try:
        dados_log, _, _ = load_data()
        
        viz = LogisticsVisualizer(dados_log)
        viz.plot_margin_evolution()
        
        print("\n✅ Teste do módulo concluído com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
