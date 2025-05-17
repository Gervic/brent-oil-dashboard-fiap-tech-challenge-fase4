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
st.title("🛢️ Dashboard de análise do preço do petróleo Brent")
st.markdown("""
Este dashboard fornece uma análise abrangente dos preços do petróleo Brent, incluindo:
- Tendência histórica
- Volatilidade dos preços
- Padrões sazonais
- Forecast do preço do próximo dia
""")

# Function to load data
@st.cache_data
def load_data():
    # Obter dados do Brent do yfinance
    ticker = "BZ=F"  # Código do Brent 
    data = yf.download(ticker, start="2010-01-01", end=datetime.now().strftime("%Y-%m-%d"))
    # Diagnóstico
    if data.empty:
        st.error("❌ Falha ao carregar dados do Yahoo Finance")
        raw_data_link = 'https://raw.githubusercontent.com/Gervic/brent-oil-dashboard-fiap-tech-challenge-fase4/refs/heads/main/petrol_price_data.csv'
        raw_data = pd.read_csv(raw_data_link, sep=';')
        brent_data = raw_data[['Date', 'petrol_price']]
        brent_data['petrol_price'] = brent_data['petrol_price'].str.replace(',', '.').astype(float)
        st.info('Dados carregados da base histórica disponível no Github')
        return brent_data
    else:
        return data

# Load the data
data = load_data()

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Tendências do preço", "Volatilidade", "Forecast"])

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

with tab1:
    st.header("Tendências do preço do petróleo Brent")

    # Lendo e preparando os dados
    try:
        df = data[['Close']].reset_index().rename(columns={'Close': 'petrol_price'})
    except:
        df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df = df.set_index('Date')

    st.sidebar.header('`Brent Oil Price Analytics`')
    st.sidebar.image("https://cdn.pixabay.com/photo/2016/03/27/22/16/pumpjack-1289654_1280.jpg", width=100)
    st.sidebar.info(f"Dados atualizados até: {df.index.max().strftime('%d/%m/%Y')}")
    ma50 = st.sidebar.slider("Média móvel curta (dias)", 10, 100, 50)
    ma200 = st.sidebar.slider("Média móvel longa (dias)", 50, 300, 200)
    show_all_events = st.sidebar.checkbox("Mostrar no gráfico todos os eventos relevantes?", value=False)

    # Cálculos
    df['volatility_30d'] = df['petrol_price'].rolling(window=30).std()
    df['ma50'] = df['petrol_price'].rolling(window=ma50).mean()
    df['ma200'] = df['petrol_price'].rolling(window=ma200).mean()
    df['price_change'] = df['petrol_price'].diff()
    df['price_pct_change'] = df['petrol_price'].pct_change() * 100
    monthly_avg = df['petrol_price'].resample('M').mean()
    yearly_avg = df['petrol_price'].resample('Y').mean()

    st.markdown("### Métricas")
    col1, col2, col3, col4 = st.columns(4)
    current_price = df['petrol_price'].iloc[-1]
    prev_price = df['petrol_price'].iloc[-2]
    pct_change = (current_price - prev_price) / prev_price * 100
    vol_30d = df['volatility_30d'].iloc[-1]

    col1.metric("Preço Atual", f"US$ {current_price:.2f}")
    col2.metric("Preço Anterior", f"US$ {prev_price:.2f}")
    col3.metric("%DoD", f"{pct_change:.2f}%")
    col4.metric("Média 30 dias", f"US$ {df['petrol_price'].tail(30).mean():.2f}")

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
    st.markdown("---")


    st.markdown("""
    Este gráfico principal mostra a trajetória completa dos preços do petróleo Brent
    ao longo de 15 anos, revelando ciclos dramáticos de alta e baixa:
    
    ● Período 2011-2014: Observamos um patamar elevado e relativamente
    estável de preços (acima de $100), sustentado pela crescente demanda
    chinesa e pelas tensões geopolíticas durante a Primavera Árabe que
    reduziram a oferta da Líbia e geraram incertezas sobre outros produtores
    da região.
    
    ● Colapso 2014-2016: Queda acentuada de preços que começou quando a
    OPEP decidiu não cortar produção em novembro de 2014, preferindo
    manter participação de mercado frente ao crescimento do xisto
    americano. A remoção das sanções contra o Irã em janeiro de 2016
    ampliou a oferta global, pressionando ainda mais os preços para baixo.
    
    ● Recuperação 2016-2018: Período de estabilização e recuperação gradual
    após o histórico Acordo da OPEP de novembro de 2016, quando o cartel
    concordou em cortar produção pela primeira vez desde 2008, em
    coordenação com produtores não-OPEP, como a Rússia.
    
    ● Choque pandêmico 2020: O colapso mais dramático da série, quando a
    conjunção da Pandemia COVID-19 e a Guerra de Preços entre Rússia e
    Arábia Saudita levou a uma queda sem precedentes, chegando ao ponto
    do WTI americano registrar preços negativos em abril de 2020.
    
    ● Recuperação pós-pandemia 2020-2022: Forte ascensão a partir de níveis
    extremamente baixos, impulsionada pela reabertura econômica global,
    pela disciplina de produção da OPEP+ e pelos pacotes de estímulo que
    fomentaram a demanda.
    
    ● Crise energética e Guerra na Ucrânia 2021-2022: A Invasão da Ucrânia
    pela Rússia em fevereiro de 2022 elevou os preços a patamares próximos
    de $130, refletindo riscos de oferta do segundo maior exportador mundial.
    Anteriormente, já havia pressão de alta devido à Crise Energética que
    elevou a demanda por petróleo como substituto do gás natural
    """)

with tab2:
    st.header("Volatilidade do preço do petróleo Brent")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('#### Volatilidade')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['volatility_30d'],
            mode='lines',
            name='30-Day Volatility',
            line=dict(color='#E74C3C')
        ))
        
        fig.update_layout(
            title="Desvio padrão móvel de 30 dias",
            xaxis_title="Data",
            yaxis_title="Volatilidade",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('#### Preços de acordo com as altas/baixas do mercado')
        monthly_avg = df[petrol_price].resample('M').mean()
        def identify_market_phases(series, threshold=0.2):
          bull_markets = []
          bear_markets = []
          
          # Initialize variables
          in_bull = False
          in_bear = False
          start_idx = 0
          peak = series.iloc[0]
          trough = series.iloc[0]
          
          for i in range(1, len(series)):
              current_price = series.iloc[i]
              
              # Check for bull market
              if not in_bull and current_price >= trough * (1 + threshold):
                  if in_bear:
                      # End of bear market
                      bear_markets.append((start_idx, i-1, series.index[start_idx], series.index[i-1], 
                                            peak, trough, (trough - peak) / peak))
                      in_bear = False
                  
                  # Start of bull market
                  in_bull = True
                  start_idx = i
                  trough = current_price
              
              # Check for bear market
              elif not in_bear and current_price <= peak * (1 - threshold):
                  if in_bull:
                      # End of bull market
                      bull_markets.append((start_idx, i-1, series.index[start_idx], series.index[i-1], 
                                          trough, peak, (peak - trough) / trough))
                      in_bull = False
                  
                  # Start of bear market
                  in_bear = True
                  start_idx = i
                  peak = current_price
              
              # Update peak and trough
              if in_bull and current_price > peak:
                  peak = current_price
              elif in_bear and current_price < trough:
                  trough = current_price
          
          # Handle the last phase
          if in_bull:
              bull_markets.append((start_idx, len(series)-1, series.index[start_idx], series.index[-1], 
                                  trough, peak, (peak - trough) / trough))
          elif in_bear:
              bear_markets.append((start_idx, len(series)-1, series.index[start_idx], series.index[-1], 
                                  peak, trough, (trough - peak) / peak))
          
          return bull_markets, bear_markets
            
        # Use monthly average for market phase identification to reduce noise
        bull_markets, bear_markets = identify_market_phases(monthly_avg, threshold=0.2)
        
        # Plot bull and bear markets
        plt.figure(figsize=(15, 7))
        plt.plot(monthly_avg.index, monthly_avg, linewidth=1, color='gray')
        
        # Highlight bull markets
        for i, (start_idx, end_idx, start_date, end_date, start_price, end_price, pct_change) in enumerate(bull_markets):
            plt.axvspan(start_date, end_date, alpha=0.2, color='green')
        
        # Highlight bear markets
        for i, (start_idx, end_idx, start_date, end_date, start_price, end_price, pct_change) in enumerate(bear_markets):
            plt.axvspan(start_date, end_date, alpha=0.2, color='red')
        
        plt.title('Oil Price with Bull (Green) and Bear (Red) Markets')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(years)
        plt.gca().xaxis.set_major_formatter(years_fmt)
        plt.xticks(rotation=45)
        st.pyplot(fig)

with tab3:
    st.header("Price Forecast")
    

# Add footer
st.markdown("---")
st.markdown("Data source: Yahoo Finance (BZ=F)") 
