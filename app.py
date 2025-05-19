import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import requests
import pickle
import joblib
import os

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
tab1, tab2, tab, tab3 = st.tabs(["Tendências do preço", "Volatilidade", "Insights", "Forecast"])

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
        df = data['Close'].reset_index().rename(columns={'BZ=F': 'petrol_price'})
    except:
        df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df = df.set_index('Date')

    st.sidebar.header('`Brent Oil Price Analytics`')
    st.sidebar.image("brent-oil-image.jpg", width=300)
    st.sidebar.info(f"Dados atualizados até: {df.index.max().strftime('%d/%m/%Y')}")
    ma50 = st.sidebar.slider("Média móvel curta (dias)", 10, 100, 50)
    ma200 = st.sidebar.slider("Média móvel longa (dias)", 50, 300, 200)
    
     # Selecionar tema
    st.sidebar.subheader("Configurações")
    show_all_events = st.sidebar.checkbox("Mostrar no gráfico todos os eventos relevantes?", value=False)
    theme = st.sidebar.selectbox("Tema", ["Claro", "Escuro"], index=0)
    
    # Aplicando tema
    if theme == "Claro":
        st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Informações adicionais
    with st.sidebar.expander("Sobre"):
        st.markdown("""
        **Brent Price Oil Analytics** é uma plataforma de análise e previsão do preço do petróleo Brent.
        
        Desenvolvida para auxiliar na tomada de decisões estratégicas baseadas em dados históricos e tendências futuras.
        
        📧 [Contato para Suporte](email para:suporte@brent-analytics.com)
        """)

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
    # Adicionando eventos importantes, se solicitado
    if show_all_events and events:
        for date, info in events.items():
            event_date = pd.to_datetime(date)
            if event_date in df.index or (event_date >= df.index[0] and event_date <= df.index[-1]):
                # Encontrar o valor y mais próximo para o evento
                closest_date = df.index[df.index.get_indexer([event_date], method='nearest')[0]]
                y_value = df.loc[closest_date, 'petrol_price']
                
                # Adicionar anotação do evento
                fig.add_annotation(
                    x=event_date,
                    y=y_value,
                    text=info['event'],
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    bgcolor="black",
                    opacity=0.8
                )
    
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
 
    df_monthly = df.copy()
    df_monthly["month"] = df_monthly.index.month
    df_monthly["year"] = df_monthly.index.year
    
    # Boxplot da sazonalidade mensal com Plotly Express
    fig = px.box(df_monthly, x="month", y="petrol_price", points="outliers",
                 labels={"month": "Mês", "petrol_price": "Preço (USD)"},
                 category_orders={"month": list(range(1, 13))},
                 title="Sazonalidade Mensal dos Preços do Petróleo Brent (2010-2025)")
    
    # Cálculo da média mensal
    monthly_means = df_monthly.groupby("month")["petrol_price"].mean()
    
    # Adiciona linha de médias mensais
    fig.add_trace(go.Scatter(
        x=list(range(1, 13)),
        y=monthly_means.values,
        mode="lines+markers",
        line=dict(color="red", width=3),
        marker=dict(size=8),
        name="Média Mensal"
    ))
    
    # Adiciona anotações
    fig.add_annotation(
        x=12,
        y=monthly_means[12],
        text="Maior demanda por<br>aquecimento<br>Hemisfério Norte",
        showarrow=True,
        arrowhead=1,
        ax=30,
        ay=-30,
        bgcolor="black"
    )
    
    fig.add_annotation(
        x=7,
        y=monthly_means[7],
        text="Temporada de<br>viagens de verão<br>Hemisfério Norte",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=50,
        bgcolor="black"
    )
    
    # Customizações de layout
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                      'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        ),
        yaxis_title="Preço (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    
    # Mostrar no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    A análise de sazonalidade revela padrões mensais que persistem apesar da alta
    volatilidade do mercado:
    
    ● Inverno no Hemisfério Norte (novembro-fevereiro): Tendência de preços
    mais altos devido à maior demanda para aquecimento, que complementa
    o consumo regular para transporte e outros usos.
    
    ● Verão no Hemisfério Norte (junho-agosto): Leve aumento de preços
    associado à temporada de viagens, quando aumenta o consumo de
    combustíveis para transporte.
    
    ● Transições sazonais (março-abril e setembro-outubro): Períodos de
    relativa fraqueza de preços, quando a demanda sazonal diminui.
    
    A análise boxplot mostra também a alta variabilidade dentro de cada mês,
    indicando que fatores fundamentais e geopolíticos frequentemente superam os
    padrões sazonais.
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
        st.markdown("""
        O gráfico de volatilidade revela períodos de maior incerteza e instabilidade no
        mercado:
        
        ● 2022: O ano com maior volatilidade média da série histórica, impulsionado
        pela Invasão da Ucrânia e subsequentes sanções ocidentais ao petróleo
        russo, além da intervenção dos EUA com a Liberação de Reservas
        Estratégicas tentando conter a alta de preços.
        
        ● 2020: Segundo ano mais volátil, dominado pelo choque da Pandemia
        COVID-19 e pela Guerra de Preços entre Arábia Saudita e Rússia,
        resultando em uma combinação catastrófica de colapso de demanda e
        aumento de oferta.
        
        ● 2011-2012: Pico de volatilidade associado à Primavera Árabe e interrupções
        de fornecimento na Líbia, Síria e outros países produtores da região.
        
        ● 2014-2016: Alta volatilidade durante o colapso de preços provocado pela
        estratégia da OPEP de não cortar produção e pelo excesso de oferta
        global.
        
        A análise deste gráfico mostra claramente que os choques geopolíticos e as
        mudanças abruptas de política dos principais produtores são os maiores
        causadores de volatilidade no mercado petrolífero.
        """)
    with c2:
        st.markdown('#### Preços de acordo com as altas/baixas do mercado')
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
        
        # Criando gráfico com Plotly
        fig = go.Figure()
        
        # Linha do preço
        fig.add_trace(go.Scatter(x=monthly_avg.index, y=monthly_avg.values,
                                 mode='lines', name='Preço Médio Mensal', line=dict(color='gray')))
        
        # Regiões de Bull Markets (verde)
        for start_idx, end_idx, start_date, end_date, *_ in bull_markets:
            fig.add_vrect(x0=start_date, x1=end_date, fillcolor="green", opacity=0.2,
                          line_width=0, annotation_text="Alta", annotation_position="top left")
        
        # Regiões de Bear Markets (vermelho)
        for start_idx, end_idx, start_date, end_date, *_ in bear_markets:
            fig.add_vrect(x0=start_date, x1=end_date, fillcolor="red", opacity=0.2,
                          line_width=0, annotation_text="Baixa", annotation_position="top left")
        
        # Layout
        fig.update_layout(
            title="Ciclos de Alta (verde) e Baixa (vermelho) - Preço do Petróleo Brent",
            xaxis_title="Data",
            yaxis_title="Preço (USD)",
            hovermode="x unified",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        Este gráfico identifica períodos distintos de mercados em alta (verde) e baixa
        (vermelho), definidos como movimentos de pelo menos 20% nos preços:
        
        ● Ciclo de Alta 2020-2022: O mais expressivo da série (+183,8%), indo de
        43,24 dolares em julho de 2020 a 122,71 dolares em agosto de 2022. 
        Este aumento foi impulsionado pela combinação de recuperação da
        demanda pós-pandemia, cortes de produção da OPEP+ e o choque da
        invasão da Ucrânia.
        
        ● Ciclo de Baixa 2020: A queda mais acentuada e rápida (-66,8%), ocorrida
        entre fevereiro e abril de 2020, quando a Pandemia COVID-19 e a Guerra
        de Preços causaram disrupção sem precedentes.
        
        ● Ciclo de Alta 2016-2018: Um período prolongado de recuperação (+75,0%)
        que durou 30 meses, começando após o acordo histórico da OPEP+ e
        sustentado pelo crescimento econômico global sincronizado.
        
        ● Ciclo de Baixa 2014-2016: Duas quedas consecutivas e severas (-44,6% e
        -49,2%) durante um período de 14 meses, quando o mercado ajustou-se
        ao excesso de oferta do xisto americano e à decisão da OPEP de priorizar
        a participação de mercado sobre preços.
        
        Este gráfico demonstra como os ciclos de petróleo tendem a ser assimétricos: as
        quedas geralmente são mais rápidas e acentuadas do que as recuperações, que
        costumam ser mais graduais.
        """)
with tab:
    # Carregando eventos importantes e insights
    def get_events_insights():
        """Retorna eventos importantes e seus insights"""
        return {
            'covid': {
                'title': 'Pandemia de COVID-19 (2020)',
                'date': '2020-03-11',
                'event_end': '2020-06-01',
                'description': '''
                A pandemia de COVID-19 causou o maior choque de preço observado nos dados recentes, com queda de 66,8% entre fevereiro e abril de 2020.
                
                **Causas principais:**
                - Lockdowns globais reduziram drasticamente a demanda por transporte
                - Guerra de preços entre Rússia e Arábia Saudita agravou a situação
                - Capacidade de armazenamento global chegou próxima ao limite
                
                Em 22 de abril de 2020, vimos uma variação diária extraordinária de +51%, refletindo a extrema volatilidade do mercado.
                A queda nos preços levou o petróleo WTI americano a registrar preços negativos pela primeira vez na história.
                ''',
                'date_range': ['2020-01-01', '2020-07-31']
            },
            'recovery': {
                'title': 'Recuperação Pós-Pandemia (2020-2022)',
                'date': '2020-07-31',
                'event_end': '2022-08-31',
                'description': '''
                A mais expressiva alta identificada na série histórica ocorreu entre julho de 2020 e agosto de 2022, com valorização de 183,8%.
                
                **Fatores de impulso:**
                - Reabertura das economias globais
                - Cortes de produção coordenados pela OPEP+
                - Retomada da demanda enquanto a oferta permanecia restrita
                - Tensões geopolíticas crescentes
                
                Este período demonstrou como os preços do petróleo podem se recuperar rapidamente após um choque, especialmente quando há ação coordenada entre os principais produtores.
                ''',
                'date_range': ['2020-07-01', '2022-08-31']
            },
            'crisis2014': {
                'title': 'Crise Financeira de 2014-2016',
                'date': '2014-11-27',
                'event_end': '2016-02-29',
                'description': '''
                Entre outubro de 2014 e fevereiro de 2016, o preço do petróleo sofreu uma queda prolongada e significativa, com duas fases distintas:
                - Primeira queda de 44,6% (out/2014 a abr/2015)
                - Segunda queda de 49,2% (jun/2015 a fev/2016)
                
                **Este período foi marcado por:**
                - Excesso de oferta devido ao boom do xisto nos EUA
                - Desaceleração da economia chinesa
                - Decisão da OPEP de não reduzir a produção para defender participação de mercado
                - Levantamento das sanções contra o Irã, aumentando a oferta global
                
                A queda prolongada levou a grandes cortes de investimentos em exploração e produção, configurando as bases para a recuperação dos preços nos anos seguintes.
                ''',
                'date_range': ['2014-10-01', '2016-03-31']
            },
            'ukraine': {
                'title': 'Guerra Russo-Ucraniana (2022)',
                'date': '2022-02-24',
                'event_end': '2022-06-30',
                'description': '''
                O ano de 2022 apresentou a maior volatilidade da série histórica. A invasão da Ucrânia pela Rússia em fevereiro de 2022 impulsionou o preço do petróleo a máximas próximas de $130.
                
                **Fatores que impactaram o mercado:**
                - Temores de sanções aos suprimentos russos (segundo maior exportador mundial)
                - Preocupações com a segurança energética europeia
                - Interrupções na infraestrutura de transporte no Mar Negro
                
                Em resposta à alta, os EUA anunciaram em março a maior liberação de reservas estratégicas da história, liberando 180 milhões de barris para tentar conter os preços.
                ''',
                'date_range': ['2022-01-01', '2022-08-31']
            },
            'arab_spring': {
                'title': 'Primavera Árabe (2011)',
                'date': '2011-03-15',
                'event_end': '2011-08-31',
                'description': '''
                Em 2011, identificamos um período de alta volatilidade e preços elevados, com o petróleo alcançando $126,64 em maio de 2011. 
                
                **Fatores de impacto:**
                - Revoltas políticas no Oriente Médio e Norte da África (Primavera Árabe)
                - Interrupção da produção líbia (perda de aproximadamente 1,6 milhão de barris por dia)
                - Temores de contágio da instabilidade para outros produtores importantes da região
                
                Este evento demonstrou como instabilidade política em regiões produtoras chave pode rapidamente elevar os preços, mesmo sem grande interrupção da oferta global.
                ''',
                'date_range': ['2011-01-01', '2011-12-31']
            },
            'energy_transition': {
                'title': 'Transição Energética e Padrões Sazonais',
                'date': '',
                'event_end': '',
                'description': '''
                A análise dos dados revela que a volatilidade do petróleo está aumentando nas últimas décadas, apesar de períodos de relativa estabilidade.
                
                **Fatores relevantes:**
                - Mudança gradual para energias renováveis alterando o balanço tradicional de oferta e demanda
                - Crescente papel de eventos climáticos extremos na determinação dos preços
                - Padrão sazonal moderado, com preços geralmente mais altos no inverno do hemisfério norte e queda na primavera
                
                A transição energética global está criando um novo paradigma para o mercado de petróleo, com investimentos reduzidos em novos projetos e maior incerteza sobre a demanda futura.
                ''',
                'date_range': ['2010-01-01', '2025-05-02']
            }
        }

    def ensure_datetime_scalar(value):
        """Converte datas para um datetime.datetime escalar seguro para uso no Plotly"""
        value = pd.to_datetime(value)
        if isinstance(value, (pd.Series, pd.Index)):
            value = value[0]  # ou .iloc[0] se for Series
        return value.to_pydatetime() if hasattr(value, 'to_pydatetime') else value
    
    # Carregando eventos
    events_insights = get_events_insights()
    
    # Título da página
    st.header("🔍 Insights Geopolíticos e Econômicos")
    st.markdown("""
    Explore os principais eventos que impactaram o mercado de petróleo e compreenda as conexões
    entre situações geopolíticas, crises econômicas e a dinâmica de preços do petróleo Brent.
    """)
    
    # Seleção de insights
    st.subheader("Selecione um evento para análise detalhada")
    
    # Criando uma visão geral dos eventos em linha do tempo
    fig = go.Figure()
    
    # Adicionando linha para o preço
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['petrol_price'],
        mode='lines',
        name='Preço Brent',
        line=dict(color='royalblue', width=2)
    ))
    
    # Adicionando marcadores para eventos importantes
    for key, event in events_insights.items():
        if event['date']:  # Verifica se há uma data específica
            event_date = pd.to_datetime(event['date'])
            if event_date in df.index or (event_date >= df.index[0] and event_date <= df.index[-1]):
                # Encontrar valor mais próximo
                closest_idx = df.index.get_indexer([event_date], method='nearest')[0]
                price = df['petrol_price'].iloc[closest_idx]
                
                fig.add_trace(go.Scatter(
                    x=[event_date],
                    y=[price],
                    mode='markers+text',
                    name=event['title'],
                    text=[event['title']],
                    textposition="top center",
                    marker=dict(size=12, symbol='circle', color='red'),
                    textfont=dict(size=10),
                    hoverinfo='text',
                    hovertext=event['title']
                ))
    
    # Formatando o gráfico
    fig.update_layout(
        title="Linha do Tempo de Eventos Importantes no Mercado de Petróleo",
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Criando um seletor de eventos
    event_options = list(events_insights.keys())
    event_titles = [events_insights[key]['title'] for key in event_options]
    selected_event_title = st.selectbox("Escolha um evento para análise detalhada:", event_titles)
    
    # Encontrando o evento selecionado
    selected_event_key = event_options[event_titles.index(selected_event_title)]
    selected_event = events_insights[selected_event_key]
    
    # Exibindo detalhes do evento selecionado
    st.header(selected_event['title'])
    st.markdown(selected_event['description'])
    
    # Filtrando dados para o período do evento
    if selected_event['date_range']:
        start_date = pd.to_datetime(selected_event['date_range'][0])
        end_date = pd.to_datetime(selected_event['date_range'][1])
        event_df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        
        # Gráfico detalhado do período do evento
        fig = go.Figure()
        
        # Adicionando linha para o preço
        fig.add_trace(go.Scatter(
            x=event_df.index,
            y=event_df['petrol_price'],
            mode='lines',
            name='Preço Brent',
            line=dict(color='royalblue', width=2)
        ))
        
        # Destacando data do evento se existir
        if selected_event['date']:
            event_date = ensure_datetime_scalar(selected_event['date'])
            fig.add_vline(
                x=event_date, 
                line_width=2, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Início: {event_date.strftime('%d/%m/%Y')}",
                annotation_position="top right"
            )
        
        # Destacando fim do evento se existir
        if selected_event['event_end']:
            event_end_date = ensure_datetime_scalar(selected_event['date'])
            fig.add_vline(
                x=event_end_date, 
                line_width=2, 
                line_dash="dash", 
                line_color="green",
                annotation_text=f"Fim: {event_end_date.strftime('%d/%m/%Y')}",
                annotation_position="top left"
            )
        
        # Formatando o gráfico
        fig.update_layout(
            title=f"Impacto do Evento: {selected_event['title']}",
            xaxis_title="Data",
            yaxis_title="Preço (USD)",
            height=500,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas relevantes
        with st.expander("Estatísticas do Período"):
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculando variação no período
            start_price = event_df['petrol_price'].iloc[0]
            end_price = event_df['petrol_price'].iloc[-1]
            price_change = end_price - start_price
            pct_change = (price_change / start_price) * 100
            
            with col1:
                st.metric("Preço Inicial", f"US$ {start_price:.2f}")
            
            with col2:
                st.metric("Preço Final", f"US$ {end_price:.2f}", f"{pct_change:.2f}%")
            
            with col3:
                st.metric("Preço Máximo", f"US$ {event_df['petrol_price'].max():.2f}")
            
            with col4:
                st.metric("Volatilidade", f"{event_df['volatility_30d'].mean():.2f}")
            
            # Exibindo dias com maior variação
            st.subheader("Dias com Maior Variação")
            top_changes = event_df.sort_values(by='price_pct_change', ascending=False).head(5)
            
            if not top_changes.empty:
                # Criando DataFrame para exibição
                display_df = pd.DataFrame({
                    'Data': top_changes.index,
                    'Preço (USD)': top_changes['petrol_price'].round(2),
                    'Variação (USD)': top_changes['price_change'].round(2),
                    'Variação (%)': top_changes['price_pct_change'].round(2)
                })
                
                st.dataframe(display_df, use_container_width=True)
    
    # Insights adicionais baseados na análise completa
    st.header("Conclusões e Análises Adicionais")
    
    # Média anual com anotações
    st.subheader("Média Anual de Preços (2010-2025)")
    
    # Calculando médias anuais
    yearly_avg = df.resample('Y')['petrol_price'].mean()
    yearly_volatility = df.resample('Y')['volatility_30d'].mean()
    
    # Criando DataFrame para visualização
    yearly_df = pd.DataFrame({
        'Ano': yearly_avg.index.year,
        'Preço Médio': yearly_avg.values,
        'Volatilidade Média': yearly_volatility.values
    })
    
    # Gráfico de barras para médias anuais
    fig = go.Figure()
    
    # Adicionando barras para preços médios
    fig.add_trace(go.Bar(
        x=yearly_df['Ano'],
        y=yearly_df['Preço Médio'],
        name='Preço Médio',
        marker_color='royalblue'
    ))
    
    # Adicionando linha para volatilidade
    fig.add_trace(go.Scatter(
        x=yearly_df['Ano'],
        y=yearly_df['Volatilidade Média'],
        mode='lines+markers',
        name='Volatilidade',
        yaxis='y2',
        line=dict(color='firebrick', width=2)
    ))
    
    # Atualizando layout para eixo duplo
    fig.update_layout(
        title="Preço Médio e Volatilidade Anual do Petróleo Brent",
        xaxis_title="Ano",
        yaxis_title="Preço Médio (USD)",
        yaxis2=dict(
            title="Volatilidade",
            overlaying="y",
            side="right"
        ),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlação com eventos econômicos mundiais
    st.subheader("Correlação com Eventos Econômicos Mundiais")
    
    st.info("""
    **Observações importantes sobre a correlação de preços do petróleo com eventos globais:**
    
    1. **Crises Financeiras**: Grandes crises financeiras, como a de 2008 (anterior ao nosso dataset) e o início da pandemia em 2020, geralmente levam a quedas acentuadas nos preços devido à redução da atividade econômica global.
    
    2. **Tensões Geopolíticas**: Conflitos em regiões produtoras (como Oriente Médio) ou envolvendo grandes produtores (como a Rússia) tendem a elevar os preços rapidamente devido ao risco de interrupção de fornecimento.
    
    3. **Decisões da OPEP+**: As decisões do cartel continuam sendo o fator individual mais importante para tendências de preços de médio prazo, como visto nas decisões de 2014, 2016 e 2020.
    
    4. **Ciclos Econômicos**: Os preços do petróleo tendem a seguir ciclos econômicos globais, com períodos de crescimento econômico sincronizado levando a aumentos de preços (como em 2017-2019 e 2021-2022).
    
    5. **Transição Energética**: O avanço das energias renováveis e políticas de descarbonização estão começando a influenciar as perspectivas de longo prazo para os preços do petróleo, potencialmente limitando picos de preço sustentados.
    """)
    
    # Conclusão
    st.header("Conclusão")
    
    st.success("""
    A análise histórica dos preços do petróleo Brent revela um mercado extremamente sensível a eventos geopolíticos, decisões de grandes produtores e mudanças macroeconômicas globais. Os principais insights obtidos são:
    
    1. O petróleo continua sendo uma commodity estratégica cujo preço reflete tensões geopolíticas globais
    2. A volatilidade do mercado tem aumentado nos últimos anos, com eventos extremos tornando-se mais frequentes
    3. Existe uma correlação clara entre decisões coordenadas de produção (OPEP+) e tendências de preços de médio prazo
    4. Grandes crises globais (como a pandemia) podem causar disrupções sem precedentes no equilíbrio de oferta e demanda
    5. O mercado apresenta uma capacidade notável de recuperação após choques, como visto na recuperação pós-pandemia
    6. A transição energética está começando a introduzir novos fatores estruturais que moldarão o mercado nas próximas décadas
    
    Estes insights são valiosos para investidores, formuladores de políticas e empresas do setor de energia que precisam navegar um ambiente cada vez mais complexo e volátil.
    """)

with tab3:
    st.header("Previsao do Preço do Petroleo Brent")
    st.markdown("### Métricas do modelo")
    cl1, cl2 = st.columns(2)
    rmse = 6.341757
    mae = 4.923904
    cl1.metric("Erro quadrático médio (RMSE)", f"US$ {rmse:.2f}")
    cl2.metric("Erro médio absoluto (MAE)", f"US$ {mae:.2f}")
    
    @st.cache_resource
    def load_model():
        return joblib.load('prophet_model.pkl')
        
    model = load_model()
    days = st.number_input("Quantos dias para prever?", min_value=1, max_value=365, value=7)
    
    future_dates = model.make_future_dataframe(periods=days)
    
    # Gerar previsão
    forecast = model.predict(future_dates)
    
    # Exibir resultado
    st.write("Previsão do preço em US$ para os próximos {} dias:".format(days))
    st.write(forecast[['ds', 'yhat']].rename(columns={'ds': 'data', 'yhat':'preço predito'}).tail(days))
    df_price = df['2025-04-01':]
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].query(" ds >= '2025-04-01' ")
    
    # Plotar previsão
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_price.index, y=df_price['petrol_price'], mode='lines', name='Histórico'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Limite superior', line=dict(dash='dot'), opacity=0.3))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Limite inferior', line=dict(dash='dot'), opacity=0.3))
    
    fig.update_layout(title="Previsão com Prophet", xaxis_title="Data", yaxis_title="Valor", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    

# Add footer
st.markdown("---")
st.markdown("Data source: Yahoo Finance (BZ=F)") 
