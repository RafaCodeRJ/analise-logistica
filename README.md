# 📊 Análise Logística Profissional

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-success.svg)

Sistema completo de análise de dados logísticos com visualizações interativas, modelagem preditiva e geração de insights acionáveis.

---

## 🎯 Objetivo

Este projeto fornece uma análise abrangente de operações logísticas, identificando:
- 📈 Tendências de rentabilidade
- 💰 Oportunidades de redução de custos
- 🚚 Eficiência operacional
- 🎯 Ações prioritárias para otimização

---

## 📁 Estrutura do Projeto

```
analise-logistica/
│
├── data/                          # Dados de entrada
│   ├── dados_logistica.csv       # Dados diários
│   ├── dados_mensais.csv         # Dados mensais consolidados
│   └── tabela_acoes.csv          # Plano de ações
│
├── notebooks/                     # Notebooks Jupyter
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_analise_preditiva.ipynb
│   └── 03_dashboard_executivo.ipynb
│
├── src/                          # Código-fonte Python
│   ├── __init__.py
│   ├── data_loader.py           # Carregamento de dados
│   ├── analyzer.py              # Análises e KPIs
│   ├── visualizer.py            # Visualizações
│   └── reporter.py              # Geração de relatórios
│
├── outputs/                      # Resultados gerados
│   ├── resumo_executivo.json
│   └── relatorio_analise.md
│
├── tests/                        # Testes unitários
│   └── test_modules.py
│
├── main.py                       # Script principal
├── criar_notebooks_profissional.py  # Gerador de notebooks
├── requirements.txt              # Dependências
├── setup.py                      # Instalação do pacote
├── .gitignore
└── README.md                     # Este arquivo
```

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo a Passo

1. **Clone o repositório:**
```bash
git clone https://github.com/RafaCodeRJ/analise-logistica.git
cd analise-logistica
```

2. **Crie um ambiente virtual (recomendado):**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Instale o pacote (opcional):**
```bash
pip install -e .
```

---

## 💻 Uso

### 1. Análise Rápida (Script Principal)

Execute o script principal para análise completa:

```bash
python main.py
```

Este script irá:
- ✅ Carregar e validar os dados
- ✅ Calcular KPIs principais
- ✅ Analisar custos e rentabilidade
- ✅ Gerar visualizações
- ✅ Criar relatórios

### 2. Notebooks Jupyter (Análise Detalhada)

#### Gerar os notebooks:
```bash
python criar_notebooks.py
```

#### Executar os notebooks:
```bash
jupyter notebook
```

Navegue até a pasta `notebooks/` e execute na ordem:
1. **01_analise_exploratoria.ipynb** - Análise exploratória completa
2. **02_analise_preditiva.ipynb** - Modelagem preditiva
3. **03_dashboard_executivo.ipynb** - Dashboard executivo

### 3. Uso Programático

```python
from src import DataLoader, LogisticsAnalyzer, LogisticsVisualizer

# Carregar dados
loader = DataLoader('data')
dados_log, dados_men, tab_acoes = loader.load_all_data()

# Análise
analyzer = LogisticsAnalyzer(dados_log)
kpis = analyzer.calculate_kpis()
print(f"Margem média: {kpis['margem_media']:.2f}%")

# Visualização
viz = LogisticsVisualizer(dados_log)
viz.plot_margin_evolution()
```

---

## 📊 Principais Funcionalidades

### 1. Análise de KPIs
- Margem de lucro (média, mediana, tendências)
- Custo por quilômetro
- Eficiência operacional
- Identificação de dias críticos

### 2. Análise de Custos
- Composição detalhada (combustível, manutenção, motorista)
- Tendências temporais
- Comparações e benchmarks

### 3. Visualizações
- Evolução temporal de métricas
- Distribuições estatísticas
- Matrizes de correlação
- Dashboards executivos

### 4. Modelagem Preditiva
- Previsão de margens
- Identificação de padrões
- Análise de importância de features
- Cenários de otimização

### 5. Plano de Ações
- Priorização por impacto
- Análise de ROI
- Roadmap de implementação

---

## 📈 Resultados Esperados

### KPIs Atuais (Exemplo)
- **Margem Média:** 28.10%
- **Custo/KM:** R$ 0.1859
- **Dias Críticos:** 30.7% do período
- **Potencial de Economia:** R$ 382.000

### Insights Principais
1. 🔴 **Tendência de queda na margem** nos últimos meses
2. 💰 **Combustível representa ~48%** dos custos
3. 📊 **Alta variabilidade** na rentabilidade diária
4. 🎯 **Ações urgentes** podem gerar R$ 127.000 de economia

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+** - Linguagem principal
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Matplotlib/Seaborn** - Visualizações
- **Scikit-learn** - Machine Learning
- **Jupyter** - Notebooks interativos
- **nbformat** - Geração de notebooks

---

## 📝 Estrutura dos Dados

### dados_logistica.csv
Dados operacionais diários:
- Data, Custos (combustível, manutenção, motorista)
- KM percorridos, Entregas, Peso transportado
- Frete, Margem, Métricas calculadas

### dados_mensais.csv
Consolidação mensal dos dados operacionais

### tabela_acoes.csv
Plano de ações com:
- ID, Descrição, Prioridade
- Impacto financeiro, Esforço, Prazo, Status

---

## 🧪 Testes

Execute os testes unitários:

```bash
python -m pytest tests/
```

Ou teste módulos individuais:

```bash
python src/data_loader.py
python src/analyzer.py
python src/visualizer.py
```

---

## 📚 Documentação Adicional

- [Guia de Instalação](docs/INSTALLATION.md)
- [Guia de Uso](docs/USAGE.md)
- [Documentação da API](docs/API.md)
- [Exemplos](docs/EXAMPLES.md)
- [FAQ](docs/FAQ.md)

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👤 Autor

**Análise Logística Profissional**

- GitHub: [@RafaCodeRJ](https://github.com/RafaCodeRJ)
- LinkedIn: [Rafael Coriolano Siqueira](https://linkedin.com/in/rafael-coriolano)

---

## 🙏 Agradecimentos

- Equipe de operações logísticas pelos dados
- Comunidade Python pela excelente documentação
- Contribuidores do projeto

---

## 📞 Suporte

Para questões e suporte:
- 📧 Email: rcoriolanosiqueira@gmail.com
- 💬 Issues: [GitHub Issues](https://github.com/RafaCodeRJ/analise-logistica/issues)

---

**⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!**
