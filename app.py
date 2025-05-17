import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="Brent Oil Price Dashboard",
    page_icon="🛢️",
    layout="wide"
)

# Title and description
st.title("🛢️ Brent Oil Price Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive analysis of Brent Oil prices, including:
- Historical price trends
- Price volatility
- Seasonal patterns
- Price forecasting
""")

# Function to load data
@st.cache_data
def load_data():
    # Get Brent Oil price data using yfinance
    ticker = "BZ=F"  # Brent Oil Future ticker
    data = yf.download(ticker, start="2010-01-01", end=datetime.now().strftime("%Y-%m-%d"))
    return data

# Load the data
data = load_data()

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Price Trends", "Volatility", "Forecast"])

#Dicionário de Eventos e Função para Anotações
# Dicionário de eventos geopolíticos e econômicos relevantes
events = {
    '2011-03-15': {'event': 'Primavera Árabe', 'desc': 'Revoltas no Oriente Médio e Norte da África'},
    '2014-11-27': {'event': 'OPEP não corta produção', 'desc': 'OPEP mantém produção apesar dos preços em queda'},
    '2016-01-16': {'event': 'Sanções do Irã removidas', 'desc': 'Fim das sanções ao Irã aumenta oferta global'},
    '2016-11-30': {'event': 'Acordo OPEP', 'desc': 'OPEP concorda em cortar produção pela primeira vez desde 2008'},
    '2019-12-06': {'event': 'OPEP+ Cortes', 'desc': 'OPEP+ aumenta cortes de produção em 500.000 barris/dia'},
    '2020-03-08': {'event': 'Guerra de Preços', 'desc': 'Arábia Saudita inicia guerra de preços após falha em acordo com Rússia'},
    '2020-03-11': {'event': 'Pandemia COVID-19', 'desc': 'OMS declara pandemia global'},
    '2020-04-20': {'event': 'WTI Negativo', 'desc': 'Preço do petróleo WTI cai para valores negativos'},
    '2021-10-04': {'event': 'Crise Energética', 'desc': 'Escassez de gás natural e carvão eleva demanda por petróleo'},
    '2022-02-24': {'event': 'Invasão da Ucrânia', 'desc': 'Rússia invade a Ucrânia'},
    '2022-03-31': {'event': 'Liberação Reservas', 'desc': 'EUA anuncia liberação de 180 milhões de barris da reserva estratégica'},
    '2023-04-02': {'event': 'Corte OPEP+', 'desc': 'OPEP+ anuncia corte surpresa de mais de 1 milhão de barris/dia'},
    '2023-10-07': {'event': 'Conflito Israel-Hamas', 'desc': 'Início do conflito entre Israel e Hamas'}
}

def add_events(ax, annotate=True, only_major=False):
    major_events = ['Primavera Árabe', 'Pandemia COVID-19', 'Invasão da Ucrânia', 'Guerra de Preços']
    
    for date, info in events.items():
        event_date = pd.to_datetime(date)
        if event_date in df.index or event_date.strftime('%Y-%m-%d') in df.index.strftime('%Y-%m-%d'):
            if only_major and info['event'] not in major_events:
                continue
            idx = df.index.get_indexer([event_date], method='nearest')[0]
            price = df.iloc[idx]['petrol_price']
            ax.axvline(x=event_date, color='gray', linestyle='--', alpha=0.7)
            if annotate:
                ax.annotate(info['event'], 
                            xy=(event_date, price),
                            xytext=(10, 40), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='black'),
                            fontsize=9, rotation=45)

with tab1:
    st.header("Brent Oil Price Trends")

    # Lendo e preparando os dados
    df = data[['Close']].reset_index().rename(columns={'Close': 'petrol_price'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df = df.set_index('Date')
    
    ma50 = st.sidebar.slider("Média móvel curta (dias)", 10, 100, 50)
    ma200 = st.sidebar.slider("Média móvel longa (dias)", 50, 300, 200)
    show_all_events = st.sidebar.checkbox("Mostrar todos os eventos?", value=False)

    # Cálculos
    df['volatility_30d'] = df['petrol_price'].rolling(window=30).std()
    df['ma50'] = df['petrol_price'].rolling(window=ma50).mean()
    df['ma200'] = df['petrol_price'].rolling(window=ma200).mean()
    df['price_change'] = df['petrol_price'].diff()
    df['price_pct_change'] = df['petrol_price'].pct_change() * 100
    monthly_avg = df['petrol_price'].resample('M').mean()
    yearly_avg = df['petrol_price'].resample('Y').mean()
    
    fig = go.Figure()

    # Preço do petróleo
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['petrol_price'],
        mode='lines',
        name='Preço Brent (USD)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Médias móveis
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ma50'],
        mode='lines',
        name=f'MM{ma50}',
        line=dict(color='green', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ma200'],
        mode='lines',
        name=f'MM{ma200}',
        line=dict(color='red', dash='dot')
    ))
    
    # Eventos (linhas verticais com anotação como hovertext)
    major_events = ['Primavera Árabe', 'Pandemia COVID-19', 'Invasão da Ucrânia', 'Guerra de Preços']
    for date_str, info in events.items():
        event_date = pd.to_datetime(date_str)
        if not show_all_events and info["event"] not in major_events:
            continue
        if df.index.min() <= event_date <= df.index.max():
            fig.add_vline(
                x=event_date,
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.7,
                annotation_text=info['event'],
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="gray"
            )
    
    # Layout
    fig.update_layout(
        title="📉 Evolução dos Preços do Petróleo Brent",
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        template="plotly_white",
        legend=dict(x=0, y=1),
        hovermode="x unified",
        height=600
    )
    
    # Mostrar no Streamlit
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Price Volatility Analysis")
    
    # Calculate daily returns and rolling volatility
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Volatility'],
        mode='lines',
        name='30-Day Volatility',
        line=dict(color='#E74C3C')
    ))
    
    fig.update_layout(
        title="30-Day Rolling Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Price Forecast")
    

# Add footer
st.markdown("---")
st.markdown("Data source: Yahoo Finance (BZ=F)") 
