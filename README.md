pip install --upgrade yfinance
pip show yfinance
pip install streamlit pandas numpy yfinance plotly scipy
pip install streamlit
pip install --upgrade streamlit-extras
!pip install streamlit-extras
!wget -q -O - ipv4.icanhazip.com
!npm install -g localtunnel@2.0.2
!streamlit run app.py & npx localtunnel --port 8501

Analisador de Investimentos Brasileiros com Streamlit
GitHub last commit
Python
Streamlit

Uma aplicação completa para análise de ações brasileiras, com foco em métricas de risco-retorno, otimização de carteiras e visualização interativa de dados.

Funcionalidades Principais
Análise de Risco-Retorno: Calcule volatilidade, Sharpe ratio, máximo drawdown e outras métricas essenciais

Otimização de Carteiras: Encontre a alocação ótima usando o modelo de Markowitz (implementado com SciPy)

Visualização Interativa: Gráficos dinâmicos com Plotly para análise de desempenho e correlação

Simulação de Monte Carlo: Explore milhares de combinações de ativos para entender a fronteira eficiente

Dados em Tempo Real: Integração direta com o Yahoo Finance para dados atualizados do mercado brasileiro
