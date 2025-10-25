"""
Módulo para carregamento e preparação dos dados logísticos.
"""

import pandas as pd
import os
from datetime import datetime

class DataLoader:
    """Classe responsável pelo carregamento dos dados."""

    def __init__(self, data_path='data'):
        """
        Inicializa o DataLoader.
        
        Args:
            data_path (str): Caminho para a pasta com os arquivos CSV
        """
        self.data_path = data_path
        self.dados_logistica = None
        self.dados_mensais = None
        self.tabela_acoes = None

    def carregar_dados(self):
        """
        Carrega todos os arquivos CSV necessários.
        
        Returns:
            tuple: (dados_logistica, dados_mensais, tabela_acoes)
        """
        try:
            # Carregar dados diários
            self.dados_logistica = pd.read_csv(
                os.path.join(self.data_path, 'dados_logistica.csv')
            )
            
            # Carregar dados mensais
            self.dados_mensais = pd.read_csv(
                os.path.join(self.data_path, 'dados_mensais.csv')
            )
            
            # Carregar tabela de ações
            self.tabela_acoes = pd.read_csv(
                os.path.join(self.data_path, 'tabela_acoes.csv')
            )
            
            # Converter coluna de data
            self.dados_logistica['Data'] = pd.to_datetime(
                self.dados_logistica['Data']
            )
            
            print("✅ Dados carregados com sucesso!")
            print(f"📅 Período analisado: {self.dados_logistica['Data'].min().strftime('%d/%m/%Y')} a {self.dados_logistica['Data'].max().strftime('%d/%m/%Y')}")
            print(f"📊 Total de registros diários: {len(self.dados_logistica)}")
            print(f"📊 Total de meses: {len(self.dados_mensais)}")
            print(f"📋 Total de ações planejadas: {len(self.tabela_acoes)}")
            
            return self.dados_logistica, self.dados_mensais, self.tabela_acoes
            
        except FileNotFoundError as e:
            print(f"❌ Erro ao carregar dados: {e}")
            print("Certifique-se de que os arquivos CSV estão na pasta 'data/'")
            raise

    def get_periodo_analise(self):
        """
        Retorna o período de análise dos dados.
        
        Returns:
            tuple: (data_inicio, data_fim)
        """
        if self.dados_logistica is not None:
            return (
                self.dados_logistica['Data'].min(),
                self.dados_logistica['Data'].max()
            )
        return None, None

    def get_estatisticas_basicas(self):
        """
        Retorna estatísticas básicas dos dados.
        
        Returns:
            dict: Dicionário com estatísticas básicas
        """
        if self.dados_logistica is None:
            return None
        
        return {
            'total_registros': len(self.dados_logistica),
            'custo_total_medio': self.dados_logistica['Custo Total'].mean(),
            'frete_medio': self.dados_logistica['Frete'].mean(),
            'margem_media': self.dados_logistica['Margem %'].mean(),
            'margem_mediana': self.dados_logistica['Margem %'].median(),
            'km_total': self.dados_logistica['KM Percorridos'].sum(),
            'entregas_total': self.dados_logistica['Entregas'].sum()
        }