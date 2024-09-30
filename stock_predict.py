import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import mplfinance as mpf

# App title
st.title("Real-Time Stock Data Dashboard with Technical Indicators and Price Prediction")

# Sidebar for user input
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, GOOGL)", "AAPL")

st.sidebar.header("Time Range")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Chart type selection
st.sidebar.header("Chart Settings")
chart_type = st.sidebar.selectbox("Select Chart Type", ["Line", "Candlestick"])

# Technical indicators
st.sidebar.header("Technical Indicators")
moving_average_period = st.sidebar.slider("Moving Average Period (days)", 5, 100, 20)
show_ma = st.sidebar.checkbox("Show Moving Average", True)

# Stock price prediction options
st.sidebar.header("Stock Price Prediction")
predict_future = st.sidebar.checkbox("Predict Future Prices")
prediction_period = st.sidebar.selectbox(
    "Select Prediction Period", 
    ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
)

# Load stock data from yfinance
@st.cache_data(ttl=300)  # Cache the data to limit API calls (refresh every 5 minutes)
def load_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    return data

# Function to train Prophet model and predict future prices
def predict_future_prices(df, period):
    # Prepare data for Prophet (remove timezone from the 'ds' column)
    prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)  # Remove timezone info
    
    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(prophet_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=period)
    
    # Forecast future prices
    forecast = model.predict(future)
    
    return forecast


# Define prediction period in days
prediction_days_map = {
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365
}

# Fetch stock data
try:
    stock_data = load_stock_data(ticker, start_date, end_date)

    # Plot stock price (with technical indicators)
    st.subheader(f"{chart_type} Chart of {ticker} from {start_date} to {end_date}")

    # If user selects candlestick chart
    if chart_type == "Candlestick":
        # Prepare data for mplfinance
        stock_data_for_mpl = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Plot candlestick chart with or without moving average
        fig, ax = plt.subplots()
        if show_ma:
            mpf.plot(stock_data_for_mpl, type='candle', mav=moving_average_period, volume=True, ax=ax)
        else:
            mpf.plot(stock_data_for_mpl, type='candle', volume=True, ax=ax)
        st.pyplot(fig)
    else:
        # If user selects line chart
        fig, ax = plt.subplots()
        stock_data['Close'].plot(ax=ax, label="Closing Price", color="blue")
        
        # Add moving average if selected
        if show_ma:
            stock_data['Moving Average'] = stock_data['Close'].rolling(window=moving_average_period).mean()
            stock_data['Moving Average'].plot(ax=ax, label=f"{moving_average_period}-Day MA", color="red")
        
        plt.legend()
        st.pyplot(fig)

    # Volume plot
    st.subheader(f"Volume of {ticker} from {start_date} to {end_date}")
    st.bar_chart(stock_data['Volume'])

    # Data summary
    st.subheader(f"Data Summary for {ticker}")
    st.dataframe(stock_data)

    # Statistics
    st.subheader(f"Statistics for {ticker}")
    st.write(stock_data.describe())

    # Prediction
    if predict_future:
        st.subheader(f"Stock Price Prediction for {ticker}")

        # Get the prediction period in days
        period_days = prediction_days_map[prediction_period]
        
        # Predict future prices using Prophet
        forecast = predict_future_prices(stock_data, period_days)

        # Plot the forecast
        st.write(f"Forecasting the next {prediction_period} for {ticker}")
        fig2 = plt.figure()
        plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Prices')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2, label='Confidence Interval')
        plt.legend()
        plt.title(f"{prediction_period} Stock Price Prediction for {ticker}")
        st.pyplot(fig2)

        # Display forecast data
        st.subheader("Predicted Data")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period_days))
    
except Exception as e:
    st.error(f"Error fetching data: {e}")

# Run with: streamlit run script_name.py
