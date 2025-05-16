import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
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
tab1, tab2, tab3, tab4 = st.tabs(["Price Trends", "Volatility", "Seasonality", "Forecast"])

with tab1:
    st.header("Brent Oil Price Trends")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#2E86C1')
    ))
    
    fig.update_layout(
        title="Historical Brent Oil Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )
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
    st.header("Seasonal Analysis")
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data['Close'], period=252)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decomposition.seasonal.index,
        y=decomposition.seasonal,
        mode='lines',
        name='Seasonal Pattern',
        line=dict(color='#27AE60')
    ))
    
    fig.update_layout(
        title="Seasonal Component of Brent Oil Prices",
        xaxis_title="Date",
        yaxis_title="Seasonal Effect (USD)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Price Forecast")
    
    # Prepare data for forecasting
    data['Target'] = data['Close'].shift(-1)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=30).std()
    
    # Drop NaN values
    data_clean = data.dropna()
    
    # Features for prediction
    features = ['Close', 'MA5', 'MA20', 'Returns', 'Volatility']
    X = data_clean[features]
    y = data_clean['Target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make prediction for next day
    last_data = X.iloc[-1:].copy()
    forecast = model.predict(last_data)[0]
    
    # Display current price and forecast
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
    with col2:
        st.metric("Next Day Forecast", f"${forecast:.2f}", 
                 delta=f"{((forecast - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%")
    
    # Plot historical prices with forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index[-30:],
        y=data['Close'].tail(30),
        mode='lines',
        name='Historical Price',
        line=dict(color='#2E86C1')
    ))
    
    fig.add_trace(go.Scatter(
        x=[data.index[-1], data.index[-1] + timedelta(days=1)],
        y=[data['Close'].iloc[-1], forecast],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#E67E22', dash='dash')
    ))
    
    fig.update_layout(
        title="Price Forecast for Next Day",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Add footer
st.markdown("---")
st.markdown("Data source: Yahoo Finance (BZ=F)") 