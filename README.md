# 📊 Análise de Dados Logísticos - Sistema de Business Intelligence

Sistema completo de análise de dados logísticos que combina análise exploratória, modelagem preditiva e dashboard executivo para otimização de operações de transporte e logística.

---

## 🚀 Visão Geral
Este projeto implementa um pipeline completo de análise de dados para operações logísticas, desde a carga de dados até a geração de insights estratégicos e recomendações baseadas em machine learning.

---

## 📈 Principais Funcionalidades
- **Análise Exploratória Completa:** Estatísticas descritivas, composição de custos, tendências temporais  
- **Modelagem Preditiva:** Previsão de margens com múltiplos algoritmos de machine learning  
- **Dashboard Executivo:** Visualização de KPIs e métricas de desempenho  

---

## 🏗️ Estrutura do Projeto
```
analise-logistica/
├── 📁 data/                          # Dados do projeto
│   ├── dados_logistica.csv          
│   ├── dados_mensais.csv            
│   └── tabela_acoes.csv             
├── 📁 src/                          
│   ├── data_loader.py              
│   ├── analise_exploratoria.py     
│   ├── analise_preditiva.py        
│   └── dashboard_executivo.py      
├── 📁 notebooks/                    
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_modelagem_preditiva.ipynb
│   └── 03_dashboard_executivo.ipynb
├── 📁 output/                       
│   ├── 📁 graficos/                
│   ├── 📁 modelos/                 
│   └── 📁 relatorios/              
├── main.py                          
├── requirements.txt                 
└── README.md                        
```

---

## 📋 Pré-requisitos
- Python 3.8+  
- pip (gerenciador de pacotes Python)

---

## ⚙️ Instalação

### Clone o repositório
```bash
git clone https://github.com/seu-usuario/analise-logistica.git
cd analise-logistica
```

### Crie um ambiente virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate   # Windows
```

### Instale as dependências
```bash
pip install -r requirements.txt
```

---

## 🎯 Como Executar

### Execução Completa
```bash
python main.py
```

### Execução por Módulos

**Análise Exploratória**
```bash
python -c "
from src.analise_exploratoria import executar_analise_exploratoria
from src.data_loader import DataLoader

loader = DataLoader('data')
dados_logistica, dados_mensais, tabela_acoes = loader.carregar_dados()
executar_analise_exploratoria(dados_logistica, dados_mensais, tabela_acoes)
"
```

**Análise Preditiva**
```bash
python -c "
from src.analise_preditiva import executar_analise_preditiva
from src.data_loader import DataLoader

loader = DataLoader('data')
dados_logistica, dados_mensais, _ = loader.carregar_dados()
executar_analise_preditiva(dados_logistica, dados_mensais)
"
```

**Dashboard Executivo**
```bash
python -c "
from src.dashboard_executivo import DashboardExecutivo
from src.data_loader import DataLoader

loader = DataLoader('data')
dados_logistica, dados_mensais, tabela_acoes = loader.carregar_dados()
dashboard = DashboardExecutivo(dados_logistica, dados_mensais, tabela_acoes)
dashboard.gerar_dashboard_completo()
"
```

---

## 📊 Resultados Gerados

**Gráficos e Visualizações:**
- 01_composicao_custos.png  
- 02_evolucao_temporal.png  
- 03_matriz_correlacao.png  
- 04_analise_dispersao.png  
- 05_importancia_features.png  
- 06_previsoes_modelo.png  
- 07_dashboard_executivo.png  
- 08_acoes_propostas.png  

**Modelos e Dados:**
- melhor_modelo.pkl  
- dados_logistica_features.csv  
- feature_importances.csv  

**Relatórios:**
- relatorio_executivo.txt  
- resumo_executivo.json  

---

## 🤖 Modelos de Machine Learning
- **Linear Regression:** Modelo linear como baseline  
- **Random Forest:** Ensemble com múltiplas árvores  
- **Gradient Boosting:** Algoritmo de boosting  

**Performance Esperada:**
- R² (teste): > 0.95  
- RMSE: < 3.0%  
- MAE: < 2.5%  

---

## 📈 KPIs e Métricas
**Financeiros:**
- Margem percentual média  
- Custo total operacional  
- Frete total  
- Composição de custos (combustível, manutenção, motorista)  

**Operacionais:**
- KM percorridos  
- Número de entregas  
- Peso transportado  
- Eficiência operacional  

**Eficiência:**
- Custo por KM  
- KM por entrega  
- Frete por KM  

---

## 🎯 Insights Estratégicos
- Identificação de Dias Críticos (margem < 20%)  
- Oportunidades de Otimização  
- Cenários de Melhoria  
- Recomendações Prioritárias  

---

## 🔧 Desenvolvimento

**Adicionando Novos Recursos**
- Novas Features → `analise_preditiva.py` → `engenharia_features()`  
- Novos Gráficos → `dashboard_executivo.py`  
- Novas Métricas → `data_loader.py` e `analise_exploratoria.py`  

**Estrutura de Dados Esperada**

`dados_logistica.csv`
```csv
Data,Custo Combustível,Custo Manutenção,Custo Motorista,Custo Total,KM Percorridos,Entregas,Peso (ton),Frete,Margem,Margem %,Custo/KM,Entregas/Dia,KM/Entrega,Mês
```

`dados_mensais.csv`
```csv
Mês,Custo Total,Custo Combustível,Custo Manutenção,Custo Motorista,Frete,Margem,KM Percorridos,Entregas,Peso (ton),Margem %,Custo/KM
```

---

## 📝 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

## 👥 Contribuição
1. Faça o fork do projeto  
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)  
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)  
4. Push para a branch (`git push origin feature/AmazingFeature`)  
5. Abra um Pull Request  

---

## 📞 Contato
Autor: Rafael Coriolano Siqueira
📧 Email: rcoriolanosiqueira@gmail.com 
🔗 LinkedIn: www.linkedin.com/in/rafael-coriolano

---

## 🚀 Próximos Passos
- Implementar API para consumo em tempo real  
- Adicionar análise de séries temporais  
- Desenvolver dashboard web interativo  
- Integrar com sistemas de gestão logística  
- Implementar alertas automáticos  

---

⭐ *Se este projeto foi útil, considere dar uma estrela no repositório!*
