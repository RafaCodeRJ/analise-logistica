#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Carregamento de Dados
================================

Responsável por carregar e validar os dados de entrada.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Classe para carregar dados logísticos."""

    def __init__(self, data_dir: str = 'data'):
        """
        Inicializa o carregador de dados.

        Args:
            data_dir: Diretório contendo os arquivos CSV
        """
        self.data_dir = Path(data_dir)
        self._validate_directory()

    def _validate_directory(self) -> None:
        """Valida se o diretório de dados existe."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Diretório não encontrado: {self.data_dir}")
        logger.info(f"✅ Diretório de dados encontrado: {self.data_dir}")

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Carrega todos os arquivos de dados.

        Returns:
            Tupla com (dados_logistica, dados_mensais, tabela_acoes)
        """
        logger.info("Carregando dados...")

        dados_logistica = self.load_dados_logistica()
        dados_mensais = self.load_dados_mensais()
        tabela_acoes = self.load_tabela_acoes()

        logger.info("✅ Todos os dados carregados com sucesso!")
        return dados_logistica, dados_mensais, tabela_acoes

    def load_dados_logistica(self) -> pd.DataFrame:
        """Carrega dados logísticos diários."""
        filepath = self.data_dir / 'dados_logistica.csv'
        df = pd.read_csv(filepath)
        df['Data'] = pd.to_datetime(df['Data'])
        logger.info(f"✅ Dados logísticos: {len(df)} registros")
        return df

    def load_dados_mensais(self) -> pd.DataFrame:
        """Carrega dados mensais consolidados."""
        filepath = self.data_dir / 'dados_mensais.csv'
        df = pd.read_csv(filepath)
        logger.info(f"✅ Dados mensais: {len(df)} registros")
        return df

    def load_tabela_acoes(self) -> pd.DataFrame:
        """Carrega tabela de ações."""
        filepath = self.data_dir / 'tabela_acoes.csv'
        df = pd.read_csv(filepath)
        logger.info(f"✅ Tabela de ações: {len(df)} registros")
        return df

    def validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Valida se o DataFrame possui as colunas necessárias.

        Args:
            df: DataFrame a ser validado
            required_columns: Lista de colunas obrigatórias

        Returns:
            True se válido, False caso contrário
        """
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"❌ Colunas faltando: {missing_columns}")
            return False
        return True


def load_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Função auxiliar para carregar dados rapidamente.

    Args:
        data_dir: Diretório contendo os arquivos CSV

    Returns:
        Tupla com (dados_logistica, dados_mensais, tabela_acoes)
    """
    loader = DataLoader(data_dir)
    return loader.load_all_data()


if __name__ == '__main__':
    # Teste do módulo
    try:
        dados_log, dados_men, tab_acoes = load_data()
        print("\n✅ Teste do módulo concluído com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
