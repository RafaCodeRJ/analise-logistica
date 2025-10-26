📊 Análise de Dados Logísticos - Sistema de Business Intelligence
Sistema completo de análise de dados logísticos que combina análise exploratória, modelagem preditiva e dashboard executivo para otimização de operações de transporte e logística.

🚀 Visão Geral
Este projeto implementa um pipeline completo de análise de dados para operações logísticas, desde a carga de dados até a geração de insights estratégicos e recomendações baseadas em machine learning.

📈 Principais Funcionalidades
- Análise Exploratória Completa: Estatísticas descritivas, composição de custos, tendências temporais

- Modelagem Preditiva: Previsão de margens com múltiplos algoritmos de machine learning

- Dashboard Executivo: Visualização de KPIs e métricas de desempenho

🏗️ Estrutura do Projeto

analise-logistica/
├── 📁 data/                          # Dados do projeto
│   ├── dados_logistica.csv          # Dados diários de operações
│   ├── dados_mensais.csv            # Dados mensais consolidados
│   └── tabela_acoes.csv             # Ações estratégicas propostas
├── 📁 src/                          # Código fonte
│   ├── data_loader.py              # Carregamento e preparação de dados
│   ├── analise_exploratoria.py     # Análise exploratória completa
│   ├── analise_preditiva.py        # Modelagem e previsão
│   └── dashboard_executivo.py      # Dashboard e relatórios
├── 📁 notebooks/                   # Jupyter notebooks
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_modelagem_preditiva.ipynb
│   └── 03_dashboard_executivo.ipynb
├── 📁 output/                      # Resultados gerados
│   ├── 📁 graficos/               # Visualizações e gráficos
│   ├── 📁 modelos/                # Modelos treinados
│   └── 📁 relatorios/             # Relatórios executivos
├── main.py                        # Script principal
├── requirements.txt               # Dependências do projeto
└── README.md                     # Documentação

📋 Pré-requisitos
- Python 3.8+
 -pip (gerenciador de pacotes Python)


⚙️ Instalação
- Clone o repositório
bash
git clone https://github.com/seu-usuario/analise-logistica.git
cd analise-logistica


Crie um ambiente virtual (recomendado):
bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows


Instale as dependências:
bash
pip install -r requirements.txt


🎯 Como Executar
Execução Completa:
bash
python main.py
Execução por Módulos


Análise Exploratória:
bash
python -c "
from src.analise_exploratoria import executar_analise_exploratoria
from src.data_loader import DataLoader

loader = DataLoader('data')
dados_logistica, dados_mensais, tabela_acoes = loader.carregar_dados()
executar_analise_exploratoria(dados_logistica, dados_mensais, tabela_acoes)
"


Análise Preditiva:
bash
python -c "
from src.analise_preditiva import executar_analise_preditiva
from src.data_loader import DataLoader

loader = DataLoader('data')
dados_logistica, dados_mensais, _ = loader.carregar_dados()
executar_analise_preditiva(dados_logistica, dados_mensais)
"


Dashboard Executivo:
bash
python -c "
from src.dashboard_executivo import DashboardExecutivo
from src.data_loader import DataLoader

loader = DataLoader('data')
dados_logistica, dados_mensais, tabela_acoes = loader.carregar_dados()
dashboard = DashboardExecutivo(dados_logistica, dados_mensais, tabela_acoes)
dashboard.gerar_dashboard_completo()
"

📊 Resultados Gerados
Gráficos e Visualizações:
01_composicao_custos.png - Composição dos custos operacionais

02_evolucao_temporal.png - Tendências temporais de margens e custos

03_matriz_correlacao.png - Correlações entre variáveis

04_analise_dispersao.png - Análise de dispersão

05_importancia_features.png - Importância das variáveis no modelo

06_previsoes_modelo.png - Comparação de previsões vs valores reais

07_dashboard_executivo.png - Dashboard com KPIs principais

08_acoes_propostas.png - Análise de ações estratégicas


Modelos e Dados:
- melhor_modelo.pkl - Melhor modelo de machine learning treinado

- dados_logistica_features.csv - Dataset com features de engenharia

- feature_importances.csv - Importância das variáveis


Relatórios:
- relatorio_executivo.txt - Relatório completo em texto

- resumo_executivo.json - Resumo em formato JSON

🤖 Modelos de Machine Learning
O sistema treina e compara múltiplos algoritmos:

- Linear Regression: Modelo linear como baseline

- Random Forest: Ensemble com múltiplas árvores

- Gradient Boosting: Algoritmo de boosting (geralmente o melhor desempenho)


Performance Esperada:

- R² (teste): > 0.95

- RMSE: < 3.0%

- MAE: < 2.5%


📈 KPIs e Métricas

Financeiros1:
- Margem percentual média

- Custo total operacional

- Frete total

- Composição de custos (combustível, manutenção, motorista)

Operacionais:
- KM percorridos

- Número de entregas

- Peso transportado

- Eficiência operacional

Eficiência:
- Custo por KM

- KM por entrega

- Frete por KM


🎯 Insights Estratégicos
O sistema gera automaticamente:

- Identificação de Dias Críticos: Dias com margem abaixo de 20%

- Oportunidades de Otimização: Redução de custos e aumento de eficiência

- Cenários de Melhoria: Simulações de impacto de ações

- Recomendações Prioritárias: Ações classificadas por impacto


🔧 Desenvolvimento
Adicionando Novos Recursos:
- Novas Features: Modifique analise_preditiva.py -> método engenharia_features()

- Novos Gráficos: Adicione métodos em dashboard_executivo.py

- Novas Métricas: Extenda data_loader.py e analise_exploratoria.py


Estrutura de Dados Esperada
Os arquivos CSV devem conter:

dados_logistica.csv:

- csv
Data,Custo Combustível,Custo Manutenção,Custo Motorista,Custo Total,KM Percorridos,Entregas,Peso (ton),Frete,Margem,Margem %,Custo/KM,Entregas/Dia,KM/Entrega,Mês

dados_mensais.csv:
- csv
Mês,Custo Total,Custo Combustível,Custo Manutenção,Custo Motorista,Frete,Margem,KM Percorridos,Entregas,Peso (ton),Margem %,Custo/KM


📝 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.


👥 Contribuição
Faça o fork do projeto

Crie uma branch para sua feature (git checkout -b feature/AmazingFeature)

Commit suas mudanças (git commit -m 'Add some AmazingFeature')

Push para a branch (git push origin feature/AmazingFeature)

Abra um Pull Request


📞 Contato
Rafael Coriolano Siqueira

Email: rcoriolanosiqueira@gmail.com

LinkedIn: www.linkedin.com/in/rafael-coriolano


🚀 Próximos Passos:
Implementar API para consumo em tempo real

Adicionar análise de séries temporais

Desenvolver dashboard web interativo

Integrar com sistemas de gestão logística

Implementar alertas automáticos


⭐ Se este projeto foi útil, considere dar uma estrela no repositório!
