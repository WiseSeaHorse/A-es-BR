# Importações essenciais
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy.optimize import minimize

# Configuração da página
st.set_page_config(layout="wide", page_title="Analisador de Investimentos Avançado", page_icon="📊")

# ==============================================
# FUNÇÕES AUXILIARES
# ==============================================

@st.cache_data(ttl=3600)
def load_data(tickers, start_date, end_date):
    """Carrega dados do Yahoo Finance com tratamento de erros"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        prices = data["Adj Close"] if 'Adj Close' in data.columns else data["Close"]
        
        if len(tickers) == 1:
            prices = prices.to_frame()
            prices.columns = [tickers[0].rstrip(".SA")]
        
        prices.columns = prices.columns.str.rstrip(".SA")
        
        # Adiciona IBOV
        ibov = yf.download("^BVSP", start=start_date, end=end_date)
        prices['IBOV'] = ibov["Adj Close"] if 'Adj Close' in ibov.columns else ibov["Close"]
        
        return prices.dropna()
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.error("Verifique os tickers digitados e o período selecionado")
        return None

def calculate_metrics(prices):
    """Calcula todas as métricas financeiras"""
    norm_prices = 100 * prices / prices.iloc[0]
    returns = prices.pct_change().dropna()
    
    # Métricas básicas
    vols = returns.std() * np.sqrt(252)
    rets = (norm_prices.iloc[-1] - 100) / 100
    sharpe_ratio = rets / vols
    
    # Métricas avançadas
    max_drawdown = (prices / prices.cummax() - 1).min()
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    
    metrics_df = pd.DataFrame({
        'Retorno': rets,
        'Volatilidade': vols,
        'Sharpe': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'VaR 95%': var_95,
        'CVaR 95%': cvar_95
    })
    
    return metrics_df, returns, norm_prices

def optimize_portfolio(returns):
    """Otimização de carteira usando scipy"""
    n_assets = len(returns.columns)
    
    # Funções para otimização
    def portfolio_return(weights):
        return np.sum(weights * returns.mean()) * 252
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    def negative_sharpe(weights):
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        return -ret / vol if vol > 0 else 0
    
    # Restrições e limites
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    
    # Otimização
    result = minimize(negative_sharpe, 
                     initial_guess,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    if result.success:
        weights = result.x
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        sharpe = -result.fun
        return weights, ret, vol, sharpe
    else:
        return None, None, None, None

def create_gauge(value, title, min_val, max_val):
    """Cria um gauge plot para métricas"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, min_val + (max_val-min_val)/3], 'color': "red"},
                {'range': [min_val + (max_val-min_val)/3, min_val + 2*(max_val-min_val)/3], 'color': "yellow"},
                {'range': [min_val + 2*(max_val-min_val)/3, max_val], 'color': "green"}
            ]
        }
    ))
    return fig

# ==============================================
# INTERFACE DO USUÁRIO
# ==============================================

# Barra lateral de configuração
with st.sidebar:
    st.header("⚙️ Configurações Avançadas")
    
    # Lista de tickers exemplo
    ticker_list = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3', 'ABEV3', 'WEGE3']
    selected_tickers = st.multiselect(
        "Selecione até 10 empresas:",
        options=ticker_list,
        default=['PETR4', 'VALE3'],
        max_selections=10
    )
    
    tickers = [t+".SA" for t in selected_tickers]
    
    # Período de análise
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Data Inicial", value=datetime(2023,1,2))
    with col2:
        end_date = st.date_input("Data Final", value="today")

# Processamento principal
if tickers:
    prices = load_data(tickers, start_date, end_date)
    
    if prices is not None:
        metrics_df, returns, norm_prices = calculate_metrics(prices)
        
        # ============= DASHBOARD RESUMO =============
        st.success("✅ Dados carregados com sucesso!")
        st.header("📌 Resumo Executivo")
        
        if len(selected_tickers) > 1:
            weights, opt_ret, opt_vol, opt_sharpe = optimize_portfolio(returns[selected_tickers])
            
            if weights is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.plotly_chart(create_gauge(opt_ret, "Retorno Esperado", -0.5, 0.5), use_container_width=True)
                with col2:
                    st.plotly_chart(create_gauge(opt_vol, "Volatilidade Esperada", 0, 1), use_container_width=True)
                with col3:
                    st.plotly_chart(create_gauge(opt_sharpe, "Sharpe Ratio Ótimo", -1, 2), use_container_width=True)
        
        # ============= ANÁLISE RISCO-RETORNO =============
        st.header("📊 Performance dos Ativos")
        
        # Cartões de métricas expandidas
        cols = st.columns(len(metrics_df))
        for i, asset in enumerate(metrics_df.index):
            with cols[i]:
                st.metric(label=asset, 
                         value=f"{metrics_df.loc[asset, 'Retorno']:.1%}",
                         delta=f"Sharpe: {metrics_df.loc[asset, 'Sharpe']:.2f}")
        
        # Gráficos avançados
        st.header("📈 Análise Gráfica")
        
        tab1, tab2 = st.tabs(["Desempenho Relativo", "Risco vs Retorno"])
        
        with tab1:
            fig = px.line(norm_prices, 
                         title="Desempenho Relativo (Base 100)",
                         labels={'value': 'Valor Normalizado', 'variable': 'Ativo'})
            st.plotly_chart(fig, use_container_width=True, height=500)
        
        with tab2:
            fig = px.scatter(
                x=metrics_df['Volatilidade'], 
                y=metrics_df['Retorno'], 
                text=metrics_df.index,
                color=metrics_df['Sharpe'],
                size=abs(metrics_df['Max Drawdown'])*100,
                color_continuous_scale=px.colors.sequential.Bluered_r,
                labels={'x': 'Volatilidade (Risco)', 'y': 'Retorno Total'},
                title='Risco vs Retorno por Ativo'
            )
            fig.update_traces(
                marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                textfont=dict(color='white', size=10)
            )
            fig.update_layout(
                height=600,
                xaxis_tickformat=".0%",
                yaxis_tickformat=".0%",
                coloraxis_colorbar=dict(title="Sharpe Ratio")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # ============= OTIMIZAÇÃO DE CARTEIRA =============
        if len(selected_tickers) > 1:
            st.header("🧮 Otimização de Carteira")
            
            weights, ret, vol, sharpe = optimize_portfolio(returns[selected_tickers])
            
            if weights is not None:
                st.success("Carteira ótima encontrada com sucesso!")
                
                weights_df = pd.DataFrame({
                    'Ativo': selected_tickers,
                    'Peso': weights
                }).sort_values('Peso', ascending=False)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Retorno Esperado", f"{ret:.1%}")
                col2.metric("Volatilidade Esperada", f"{vol:.1%}")
                col3.metric("Índice de Sharpe", f"{sharpe:.2f}")
                
                # Mostrar alocações
                fig = px.pie(weights_df, 
                             values='Peso', 
                             names='Ativo', 
                             title='Alocação Ótima da Carteira')
                st.plotly_chart(fig, use_container_width=True)
                
                # Simulação de Monte Carlo
                st.subheader("Simulação de Carteiras Aleatórias")
                n_portfolios = 10000
                results = np.zeros((3, n_portfolios))
                
                for i in range(n_portfolios):
                    w = np.random.random(len(selected_tickers))
                    w /= np.sum(w)
                    results[0,i] = np.sum(w * returns[selected_tickers].mean()) * 252
                    results[1,i] = np.sqrt(np.dot(w.T, np.dot(returns[selected_tickers].cov() * 252, w)))
                    results[2,i] = results[0,i] / results[1,i]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results[1,:], y=results[0,:], 
                    mode='markers',
                    marker=dict(
                        color=results[2,:],
                        colorscale='Viridis',
                        size=5,
                        showscale=True,
                        colorbar=dict(title='Sharpe Ratio')
                    ),
                    name='Carteiras Aleatórias'
                ))
                fig.add_trace(go.Scatter(
                    x=[vol], y=[ret],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=15,
                        symbol='star'
                    ),
                    name='Carteira Ótima'
                ))
                fig.update_layout(
                    title='Simulação de Carteiras Aleatórias',
                    xaxis_title='Volatilidade',
                    yaxis_title='Retorno',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Não foi possível otimizar a carteira. Verifique os dados.")
        
        # ============= ANÁLISE DE CORRELAÇÃO =============
        st.header("🔗 Matriz de Correlação")
        corr_matrix = returns[selected_tickers].corr() if len(selected_tickers) > 1 else returns.corr()
        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        title='Correlação entre Ativos')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Selecione empresas na barra lateral para começar")

# Rodapé
st.markdown("---")
st.caption(f"Analisador de Investimentos Avançado | Dados do Yahoo Finance | Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M')}")