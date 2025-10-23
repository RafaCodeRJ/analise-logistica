#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Geração de Relatórios
================================

Gera relatórios em diferentes formatos.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Classe para geração de relatórios."""
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Inicializa o gerador de relatórios.
        
        Args:
            output_dir: Diretório para salvar relatórios
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"✅ Diretório de saída: {self.output_dir}")
    
    def generate_executive_summary(self, kpis: Dict, custos: Dict, 
                                   classificacao: Dict, roi: Dict) -> str:
        """
        Gera resumo executivo em JSON.
        
        Args:
            kpis: Dicionário com KPIs
            custos: Dicionário com análise de custos
            classificacao: Dicionário com classificação de dias
            roi: Dicionário com análise de ROI
            
        Returns:
            Caminho do arquivo gerado
        """
        resumo = {
            'data_geracao': datetime.now().isoformat(),
            'kpis': kpis,
            'custos': custos,
            'classificacao_dias': classificacao,
            'plano_acoes': roi
        }
        
        filepath = self.output_dir / 'resumo_executivo.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(resumo, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✅ Resumo executivo salvo: {filepath}")
        return str(filepath)
    
    def generate_csv_report(self, dados: pd.DataFrame, filename: str) -> str:
        """
        Gera relatório em CSV.
        
        Args:
            dados: DataFrame com dados
            filename: Nome do arquivo
            
        Returns:
            Caminho do arquivo gerado
        """
        filepath = self.output_dir / filename
        dados.to_csv(filepath, index=False, encoding='utf-8')
        
        logger.info(f"✅ Relatório CSV salvo: {filepath}")
        return str(filepath)
    
    def generate_markdown_report(self, kpis: Dict, custos: Dict, 
                                 classificacao: Dict) -> str:
        """
        Gera relatório em Markdown.
        
        Args:
            kpis: Dicionário com KPIs
            custos: Dicionário com análise de custos
            classificacao: Dicionário com classificação de dias
            
        Returns:
            Caminho do arquivo gerado
        """
        md_content = f"""# Relatório de Análise Logística

**Data de Geração:** {datetime.now().strftime('%d/%m/%Y %H:%M')}

---

## 📊 KPIs Principais

| Métrica | Valor |
|---------|-------|
| Margem Média | {kpis['margem_media']:.2f}% |
| Margem Mediana | {kpis['margem_mediana']:.2f}% |
| Custo/KM Médio | R$ {kpis['custo_km_medio']:.4f} |
| Entregas Totais | {kpis['entregas_totais']:,.0f} |
| KM Totais | {kpis['km_totais']:,.0f} km |
| Dias Críticos | {kpis['dias_criticos']} ({kpis['percentual_dias_criticos']:.1f}%) |

---

## 💰 Composição de Custos

| Categoria | Valor | Percentual |
|-----------|-------|------------|
| Combustível | R$ {custos['custo_combustivel']:,.2f} | {custos['prop_combustivel']:.1f}% |
| Manutenção | R$ {custos['custo_manutencao']:,.2f} | {custos['prop_manutencao']:.1f}% |
| Motorista | R$ {custos['custo_motorista']:,.2f} | {custos['prop_motorista']:.1f}% |
| **TOTAL** | **R$ {custos['custo_total']:,.2f}** | **100%** |

---

## 📈 Classificação de Dias

| Classificação | Quantidade | Percentual |
|---------------|------------|------------|
| 🟢 Excelente (≥35%) | {classificacao['excelente']} | {classificacao['excelente']/sum(classificacao.values())*100:.1f}% |
| 🔵 Bom (25-35%) | {classificacao['bom']} | {classificacao['bom']/sum(classificacao.values())*100:.1f}% |
| 🟡 Regular (20-25%) | {classificacao['regular']} | {classificacao['regular']/sum(classificacao.values())*100:.1f}% |
| 🔴 Crítico (<20%) | {classificacao['critico']} | {classificacao['critico']/sum(classificacao.values())*100:.1f}% |

---

## 🎯 Recomendações

### Ações Urgentes (30 dias)
1. Renegociar contratos com fornecedores
2. Implementar controle rigoroso de combustível
3. Otimizar rotas de entrega

### Ações Prioritárias (60-90 dias)
4. Implementar sistema de previsão de demanda
5. Expandir frota estrategicamente
6. Diversificar base de clientes

### Ações de Longo Prazo (180 dias)
7. Renovar frota gradualmente
8. Implementar ERP integrado

---

**Gerado automaticamente pelo Sistema de Análise Logística**
"""
        
        filepath = self.output_dir / 'relatorio_analise.md'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"✅ Relatório Markdown salvo: {filepath}")
        return str(filepath)


if __name__ == '__main__':
    # Teste do módulo
    print("✅ Módulo de relatórios carregado com sucesso!")
