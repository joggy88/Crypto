import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf  # For fetching financial data
import feedparser
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from datetime import date
from time import sleep
import random
def load_data(ticker):
    START = "2023-01-05"
    TODAY = date.today().strftime('%Y-%m-%d')
    try:
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        # Ensure that the necessary columns are present in the DataFrame
        if 'Close' not in data.columns:
            raise ValueError(f"Column 'Close' not found in the data for {ticker}.")
        return data
    except Exception as e:
        raise ValueError(f"Failed to load data for {ticker}. Error: {str(e)}")
def create_yfinance_chart(df, selected_Crypto):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], name=f'{selected_Crypto}_Opening'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=f'{selected_Crypto}_Closing'))
    fig.update_layout(title_text=f"{selected_Crypto} Time Series Data", xaxis_rangeslider_visible=True)
    return fig
def fetch_top_stories(coin):
    rss_url = f"https://finance.yahoo.com/rss/headline?s={coin}"
    feed = feedparser.parse(rss_url)

    top_stories = []
    for entry in feed.entries:
        story = {
            "title": entry.title,
            "link": entry.link,
            "published": entry.published
        }
        top_stories.append(story)

    return top_stories

def fetch_realtime_data(ticker):
    # Fetch the most recent data for the last 1 minute interval
    realtime_data = yf.download(ticker, period='1d', interval='1m')
    return realtime_data
def fetch_historical_price_data(coin, start_date, end_date):
    # Fetch historical price data using yfinance library
    df = yf.download(coin, start=start_date, end=end_date)
    return df


def preprocess_data(price_data):
    # Convert Timestamp columns to Unix timestamps
    price_data['Timestamp'] = price_data['Timestamp'].apply(lambda x: x.timestamp())
    # ... other preprocessing steps

    # Return the processed price_data
    return price_data

def calculate_rolling_means(df, window):
    # Calculate rolling means using pandas' rolling function
    rolling_means = df['Close'].rolling(window=window).mean()
    return rolling_means

def generate_moving_average_graph(df, rolling_means):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Price')
    plt.plot(df.index, rolling_means, label='Day Rolling Mean', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Moving Average Analysis')
    plt.legend()
    st.pyplot(plt)



def calculate_correlation_matrix(coin, coin_data, other_coins_data):
    correlation_matrix = coin_data.corrwith(other_coins_data)
    return correlation_matrix


def detect_market_sentiment(price_data, rolling_means):
    last_price = price_data.iloc[-1]['Close']
    last_rolling_mean = rolling_means.iloc[-1]

    if last_price > last_rolling_mean:
        return "Bullish (Buy Signal)"
    else:
        return "Bearish (Sell Signal)"


def calculate_macd(df, short_period, long_period, signal_period):
    df['ShortEMA'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    df['LongEMA'] = df['Close'].ewm(span=long_period, adjust=False).mean()
    df['MACD'] = df['ShortEMA'] - df['LongEMA']
    df['Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    return df

def create_inputs(price_data, window_len=10):
    inputs = []
    for i in range(len(price_data) - window_len):
        temp_set = price_data[i:(i + window_len)].copy()
        inputs.append(temp_set)
        inputs[i]['Close'] = inputs[i]['Close'] / inputs[i]['Close'].iloc[0] - 1
        inputs[i]['Volume'] = inputs[i]['Volume'] / inputs[i]['Volume'].iloc[0] - 1
    return inputs



import yfinance as yf
import pandas as pd
import datetime

def fetch_historical_price_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_rolling_average(data, window_size):
    return data['Close'].rolling(window=window_size).mean()

def generate_signals(data):
    data['Signal'] = 0
    data.loc[data['Close'] > data['RollingAvg'], 'Signal'] = 1  # Buy signal
    data.loc[data['Close'] < data['RollingAvg'], 'Signal'] = -1  # Sell signal
    return data

def get_buy_signals(data):
    return data[data['Signal'] == 1]

def get_sell_signals(data):
    return data[data['Signal'] == -1]


def calculate_correlation(data1, data2):
    return data1['Close'].corr(data2['Close'])


def get_correlations(base_coin_data, coin_list):
    correlations = {}

    for coin in coin_list:
        coin_data = fetch_historical_price_data(coin, base_coin_data.index[0], base_coin_data.index[-1])
        correlation = calculate_correlation(base_coin_data, coin_data)
        correlations[coin] = correlation

    return correlations


def get_top_correlated_coins(correlations, threshold):
    correlated_coins = []
    anti_correlated_coins = []

    for coin, correlation in correlations.items():
        if correlation > threshold:
            correlated_coins.append(coin)
        elif correlation < -threshold:
            anti_correlated_coins.append(coin)

    return correlated_coins, anti_correlated_coins