import streamlit as st
import pandas as pd
from data_loader import fetch_top_stories, create_inputs, load_data, calculate_macd, detect_market_sentiment, calculate_correlation_matrix, fetch_historical_price_data, fetch_realtime_data, create_yfinance_chart, calculate_rolling_means, generate_moving_average_graph
import numpy as np
from tensorflow import keras
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from datetime import date
from time import sleep
import random
from tensorflow import keras
import model_loader
from model_loader import load_btc_lstm_model, load_eth_lstm_model, load_ltc_lstm_model
from model_loader import load_btc_rf_model, load_eth_rf_model, load_ltc_rf_model
from model_loader import load_eth_arima_model, load_ltc_arima_model, load_btc_arima_model
from data_loader import fetch_historical_price_data, get_top_correlated_coins, calculate_correlation, calculate_rolling_average, generate_signals, get_buy_signals, get_sell_signals
from data_loader import fetch_historical_price_data, get_top_correlated_coins, get_correlations
from data_loader import create_inputs
from PIL import Image
import base64
import yfinance as yf

# Function to set page style
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set page style
def set_page_style(base64_image):
    page_bg = f"""
    <style>
    body {{
        background-image: url('data:image/jpg;base64,{base64_image}');
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)
def main():
    st.set_page_config(
        page_title="Soligence Crypto Predictor",
        page_icon="📈",
        layout="wide",
    )

    # Define CSS to set the background image
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url('btc.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Replace 'btc.jpg' with the path to your image file
    base64_image = get_base64_of_bin_file('AE2/btc.jpg')

    set_page_style(base64_image)
    st.title("Soligence Crypto Predictor")
    st.warning("NOTE : Cryptos tend to fluctuate quickly so kindly invest in them at your own risk")

    # Sidebar menu
    menu = st.sidebar.selectbox("Menu", ["Home", "Cryptocurrencies", "Models", "Purpose"])

    if menu == "Home":
        st.image('AE2/btc.jpg', use_column_width=True)
        st.write("Welcome to Soligence Crypto Predictor!")
        # Display introduction or any other relevant content

    elif menu == "Cryptocurrencies":
        selected_coin = st.selectbox("Select a cryptocurrency", ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD', 'DOGE-USD', 'DOT1-USD', 'LINK-USD', 'MATIC-USD', 'XLM-USD', 'BNB-USD', 'EOS-USD', 'XMR-USD', 'TRX-USD', 'VET-USD', 'THETA-USD', 'MIOTA-USD', 'FIL-USD', 'XTZ-USD'])
        st.write(f"Top stories for {selected_coin}:")
        top_stories = fetch_top_stories(selected_coin)

        for story in top_stories:
            st.write(f"- [{story['title']}]({story['link']}) ({story['published']})")

    elif menu == "Models":
        selected_model = st.selectbox("Select a model", ["LSTM", "ARIMA", "Random Forest"])
        if selected_model == "LSTM":
            st.write("<h2 style='font-family: Arial, sans-serif; color: #3366cc;'>LSTM Model:</h2>", unsafe_allow_html=True)
            st.write(
                "Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that can capture long-range dependencies in time series data.")
            st.write("Using LSTM training process aims to minimize the difference between predicted and actual values, allowing the model to learn patterns and relationships in the data. It's an iterative process that requires finding a balance between fitting the training data well and generalizing to new data.")

            # Display the LSTM preparation code


            # Display the LSTM success metrics
            st.write("<h2 style='font-family: Arial, sans-serif; color: #3366cc;'>LSTM Success Metrics:</h2>", unsafe_allow_html=True)
            st.write("R-squared score: 78.95534187645835")
            st.write("MSE: 0.0016226153766683796")
            st.write("Mean Absolute Error: 0.0281576199339344")
            st.write("LSTM Model Actual and Prediction Graph:")
            st.image('lstm_graph.png', use_column_width=True)

        elif selected_model == "Random Forest":
            st.write("<h2 style='font-family: Arial, sans-serif; color: #3366cc;'>Random Forest Model:</h2>", unsafe_allow_html=True)
            st.write("<p style='font-family: Helvetica, sans-serif;'>Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. It is commonly used for regression and classification tasks, including time series forecasting.</p>", unsafe_allow_html=True)
            st.write("Utilizing Random Forest for cryptocurrency price prediction offers benefits in handling non-linearity and noisy data. Careful feature selection and ensemble strategies enhanced the accuracy in short-term forecasts amid market volatility..")


            # Display the Random Forest metrics and explanations
            st.write("<h2 style='font-family: Arial, sans-serif; color: #3366cc;'>Random Forest Model Success Metrics:</h2>", unsafe_allow_html=True)
            st.write("To evaluate the performance of the Random Forest model, various metrics are used:")
            st.write(f"Mean Squared Error (MSE): 0.0018")
            st.write(f"Mean Absolute Error (MAE): 0.0314")
            st.write(f"Mean Absolute Percentage Error (MAPE): 3.13%")
            st.write(f"R-squared (R²) Score: 0.7725")
            st.write("Random Forest Model Actual and Prediction Graph:")
            st.image('AE2/randomforest_graph.png', use_column_width=True)


        elif selected_model == "ARIMA":
            st.write("<h2 style='font-family: Arial, sans-serif; color: #3366cc;'>ARIMA Model:</h2>", unsafe_allow_html=True)
            st.write("ARIMA (Autoregressive Integrated Moving Average) is a classical time series forecasting method. It's used to model and forecast time series data by incorporating autoregression, differencing, and moving average components.")
            st.write(
                "ARIMA, a versatile time series forecasting method, is well-suited for this task due to its ability to capture temporal dependencies and patterns in data.")

            # Display the ARIMA metrics and explanations
            st.write("<h2 style='font-family: Arial, sans-serif; color: #3366cc;'>ARIMA Success Metrics:</h2>", unsafe_allow_html=True)
            st.write("To evaluate the performance of the ARIMA model, various metrics are used:")
            st.write(f"Mean Squared Error (MSE): 0.0011")
            st.write(f"Mean Absolute Error (MAE): 0.0522")
            st.write(f"Mean Absolute Percentage Error (MAPE): 7.23%")
            st.write(f"R-squared (R²) Score: 0.7612")
            st.write("Original Bitcoin Price Chart During Prediction:")
            st.image('AE2/BTC_graph.png', use_column_width=True)
            st.write("ARIMA Model Actual and Prediction Graph:")
            st.image('AE2/ARIMA_graph.png', use_column_width=True)



    elif menu == "Purpose":
        purpose_submenu = st.selectbox("Select a purpose", ["EDA", "Correlation", "Detection", "Prediction", "Trends"])

        if purpose_submenu == "EDA":
            st.write("Exploratory Data Analysis:")

            selected_coin = st.selectbox("Select a cryptocurrency", ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD', 'DOGE-USD', 'DOT1-USD', 'LINK-USD', 'MATIC-USD', 'XLM-USD', 'BNB-USD', 'EOS-USD', 'XMR-USD', 'TRX-USD', 'VET-USD', 'THETA-USD', 'MIOTA-USD', 'FIL-USD', 'XTZ-USD'])



            today = datetime.date.today()
            start_date = datetime.date(2017, 1, 1)

            price_data = fetch_historical_price_data(selected_coin, start_date, today)
            st.write(f"Displaying EDA for {selected_coin} from {start_date} to {today}")

            # Calculate rolling means with a chosen window size
            rolling_window = st.slider("Rolling Window Size", min_value=1, max_value=len(price_data), value=30)
            rolling_means = calculate_rolling_means(price_data, rolling_window)
            generate_moving_average_graph(price_data, rolling_means)


            st.plotly_chart(create_yfinance_chart(price_data, selected_coin))
            real_time_chart_container = st.empty()
            while True:
                real_time_data = fetch_realtime_data(selected_coin)
                if not real_time_data.empty:
                    fig = create_yfinance_chart(real_time_data, selected_coin)
                    real_time_chart_container.plotly_chart(fig, use_container_width=True)
                sleep(30)

            btc_data = fetch_historical_price_data("BTC", start_date, today)
            eth_data = fetch_historical_price_data("ETH", start_date, today)
            ltc_data = fetch_historical_price_data("LTC", start_date, today)

            # Calculate correlation matrix
            correlation_btc = calculate_correlation_matrix(selected_coin, price_data['Close'], btc_data['Close'])
            correlation_eth = calculate_correlation_matrix(selected_coin, price_data['Close'], eth_data['Close'])
            correlation_ltc = calculate_correlation_matrix(selected_coin, price_data['Close'], ltc_data['Close'])

            # Display correlation matrix for BTC
            st.write(f"Correlation matrix with {selected_coin} and BTC:")
            st.write(correlation_btc)

            # Display correlation matrix for ETH
            st.write(f"Correlation matrix with {selected_coin} and ETH:")
            st.write(correlation_eth)

            # Display correlation matrix for LTC
            st.write(f"Correlation matrix with {selected_coin} and LTC:")
            st.write(correlation_ltc)




        elif purpose_submenu == "Correlation":

            st.write("Cryptocurrency Correlation Analysis:")

            coin_symbols = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD', 'XRP-USD', 'ADA-USD', 'MATIC-USD',

                            'DOGE-USD', 'BUSD-USD']  # Define your cryptocurrency symbols

            selected_crypto = st.selectbox("Select a cryptocurrency:", coin_symbols)

            start_date = st.date_input("Select start date:", datetime.date(2017, 1, 1))

            end_date = st.date_input("Select end date:", datetime.date.today())

            correlation_data = yf.download(tickers=coin_symbols, start=start_date, end=end_date)['Close']

            # Calculate correlation matrix

            correlation_matrix = correlation_data.corr()

            # Get correlations with the selected cryptocurrency

            selected_correlations = correlation_matrix[selected_crypto].sort_values(ascending=False)

            # Display positive correlations

            positive_correlations = selected_correlations[selected_correlations > 0][:10]

            st.write("Positive Correlations:")

            st.write(positive_correlations)

            # Display negative correlations

            negative_correlations = selected_correlations[selected_correlations < 0][:10]

            st.write("Negative Correlations:")

            st.write(negative_correlations)


        elif purpose_submenu == "Trends":
            st.write("Trends Analysis:")
            selected_coin = st.selectbox("Select a cryptocurrency",
                                         ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD', 'DOGE-USD',
                                          'DOT1-USD', 'LINK-USD', 'MATIC-USD', 'XLM-USD', 'BNB-USD', 'EOS-USD',
                                          'XMR-USD', 'TRX-USD', 'VET-USD', 'THETA-USD', 'MIOTA-USD', 'FIL-USD',
                                          'XTZ-USD'])
            start_date = datetime.date(2017, 1, 1)
            today = datetime.date.today()
            price_data = fetch_historical_price_data(selected_coin, start_date, today)
            st.write(f"Displaying trends analysis for {selected_coin}")

            # Calculate MACD indicators
            short_period = st.slider("Short EMA Period (This EMA responds more quickly to recent price changes and is used to calculate the MACD line. A shorter period captures more short-term fluctuations in the price. Traders often use periods around 12 days to capture short-term trends but you can adjust to your like.)", min_value=1, max_value=50, value=12)
            long_period = st.slider("Long EMA Period (This EMA responds more slowly to price changes and is also used to calculate the MACD line. A longer period helps smooth out noise and captures longer-term trends. Typical periods might be around 26 days but adjustable)", min_value=1, max_value=200, value=26)
            signal_period = st.slider("Signal Period (The signal period refers to the number of time periods used to calculate the EMA of the MACD line itself. This EMA is known as the signal line and is used to generate buy and sell signals based on crossovers with the MACD line. A shorter signal period, such as 9 periods, is often used to make the signal line more sensitive to crossovers, you can adjust to choose your days.)", min_value=1, max_value=50, value=9)
            price_data = calculate_macd(price_data, short_period, long_period, signal_period)

            # Visualize MACD lines and crossovers
            plt.figure(figsize=(10, 6))
            plt.plot(price_data.index, price_data['MACD'], label='MACD Line', color='blue')
            plt.plot(price_data.index, price_data['Signal'], label='Signal Line', color='red')
            plt.bar(price_data.index, price_data['Histogram'], label='Histogram', color='gray')
            plt.xlabel('Date')
            plt.ylabel('MACD')
            plt.title('MACD Analysis')
            plt.legend()
            st.pyplot(plt)
            # Generate and display buy/sell signals based on trends





        elif purpose_submenu == "Detection":
            st.write("Market Detection:")

            selected_coin = st.selectbox("Select a cryptocurrency",
                                         ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD', 'DOGE-USD',
                                          'DOT1-USD', 'LINK-USD', 'MATIC-USD', 'XLM-USD', 'BNB-USD', 'EOS-USD',
                                          'XMR-USD', 'TRX-USD', 'VET-USD', 'THETA-USD', 'MIOTA-USD', 'FIL-USD',
                                          'XTZ-USD'])

            start_date = datetime.date(2017, 1, 1)
            today = datetime.date.today()

            price_data = fetch_historical_price_data(selected_coin, start_date, today)
            rolling_window_size = st.slider("Select the rolling mean window size", min_value=1, max_value=100, value=50)

            rolling_avg = calculate_rolling_average(price_data, rolling_window_size)
            price_data['RollingAvg'] = rolling_avg

            price_data_with_signals = generate_signals(price_data)
            buy_signals = get_buy_signals(price_data_with_signals)
            sell_signals = get_sell_signals(price_data_with_signals)

            st.write(f"Displaying market detection for {selected_coin}")

            possible_high = price_data['High'].max()  # Maximum high in the data
            possible_low = price_data['Low'].min()  # Minimum low in the data

            st.write(f"Estimated possible high for {selected_coin}: {possible_high:.2f}")
            st.write(f"Estimated possible low for {selected_coin}: {possible_low:.2f}")

            if not buy_signals.empty and not sell_signals.empty:
                buy_time = buy_signals.index[0]
                sell_time = sell_signals.index[-1]
                purchase_price = buy_signals['Close'].iloc[0]
                sale_price = sell_signals['Close'].iloc[-1]
                profit = sale_price - purchase_price

                st.write(f"Best time to purchase {selected_coin}: {buy_time}")
                st.write(f"Best time to sell {selected_coin}: {sell_time}")
                st.write(f"Purchase price: {purchase_price:.2f}")
                st.write(f"Selling price: {sale_price:.2f}")

                if profit > 0:
                    st.write(f"Anticipated profit: {profit:.2f}")
                else:
                    loss = abs(profit)
                    st.write(f"Anticipated loss: {loss:.2f}")
            else:
                st.write("No buy or sell signals detected.")

            # Create and display the chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], name=f'{selected_coin}_Closing'))
            fig.add_trace(go.Scatter(x=price_data.index, y=price_data['RollingAvg'], name='Rolling Average'))

            if not buy_signals.empty:
                fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                         mode='markers', name='Buy Signal', marker=dict(color='green', size=10)))

            if not sell_signals.empty:
                fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                         mode='markers', name='Sell Signal', marker=dict(color='red', size=10)))

            fig.update_layout(title_text=f"{selected_coin} Time Series Data with Buy/Sell Signals",
                              xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

            # Display buy and sell signals
            if buy_signals.empty:
                st.write("No buy signals detected.")
            else:
                st.write("Potential buy signals detected at the following dates:")
                st.write(buy_signals.index.tolist())

            if sell_signals.empty:
                st.write("No sell signals detected.")
            else:
                st.write("Potential sell signals detected at the following dates:")
                st.write(sell_signals.index.tolist())


        elif purpose_submenu == "Prediction":

            st.write("Prediction Submenu:")

            model_options = ["LSTM", "Random Forest", "ARIMA"]

            selected_model = st.selectbox("Select a model", model_options)

            if selected_model == "LSTM":

                st.write("LSTM Model:")


                coin_options = ["BTC", "ETH", "LTC"]

                selected_coin = st.selectbox("Select a coin", coin_options)

                price_data = load_data(selected_coin)

                if selected_coin == "BTC":

                    lstm_model = load_btc_lstm_model()

                elif selected_coin == "ETH":

                    lstm_model = load_eth_lstm_model()

                elif selected_coin == "LTC":

                    lstm_model = load_ltc_lstm_model()

                sell_price = st.number_input("Enter the sell price:", min_value=0.0)
                coin_quantity = st.number_input("Enter the quantity of coins:", min_value=0.0)

                if st.button("Calculate and Recommend"):

                    # Prepare input data using the create_inputs function
                    btc_input = create_inputs(price_data)

                    # Convert btc_input to a numpy array
                    btc_input = np.array(btc_input)

                    # Use the loaded LSTM model to make predictions
                    predicted_price_normalized = lstm_model.predict(btc_input)

                    predicted_price = predicted_price_normalized[0][0] * btc_input[0][-1]['Close'] + btc_input[0][-1][
                        'Close']

                    # Calculate potential profit/loss
                    potential_profit = (sell_price - predicted_price) * coin_quantity

                    # Display the investment scenario results to the user
                    st.write(f"Predicted {selected_coin} Price: {predicted_price:.2f}")
                    st.write(f"Potential Profit/Loss: ${potential_profit:.2f}")

            elif selected_model == "Random Forest":
                coin_options = ["BTC", "ETH", "LTC"]

                selected_coin = st.selectbox("Select a coin", coin_options)

                if selected_coin == "BTC":

                    rf_model = load_btc_rf_model()

                elif selected_coin == "ETH":

                    rf_model = load_eth_rf_model()

                elif selected_coin == "LTC":

                    rf_model = load_ltc_rf_model()

                sell_price = st.number_input("Enter the sell price:", min_value=0.0)

                coin_quantity = st.number_input("Enter the quantity of coins:", min_value=0.0)

                if st.button("Calculate"):
                    # Prepare input data for the Random Forest model using create_inputs function

                    today = datetime.date.today()
                    start_date = datetime.date(2017, 1, 1)

                    price_data = fetch_historical_price_data(selected_coin, start_date, today)

                    coin_data = fetch_historical_price_data(selected_coin, start_date, today)

                    input_data = create_inputs(coin_data)  # Assuming coin_data is the relevant DataFrame

                    # Convert input_data to a numpy array

                    input_data = np.array(input_data)

                    # Use the loaded Random Forest model to make predictions

                    predicted_price = rf_model.predict(input_data)

                    # Calculate potential profit/loss

                    potential_profit = (sell_price - predicted_price[-1]) * coin_quantity

                    # Display the investment scenario results to the user

                    st.write(f"Predicted {selected_coin} Price: {predicted_price[-1]:.2f}")

                    st.write(f"Potential Profit/Loss: ${potential_profit:.2f}")

            elif selected_model == "ARIMA":
                st.write("ARIMA Model:")
                st.write("ARIMA (AutoRegressive Integrated Moving Average) is a time series forecasting method.")

                coin_options = ["BTC", "ETH", "LTC"]
                selected_coin = st.selectbox("Select a coin", coin_options)

                if selected_coin == "BTC":
                    arima_model = load_btc_arima_model()
                elif selected_coin == "ETH":
                    arima_model = load_eth_arima_model()
                elif selected_coin == "LTC":
                    arima_model = load_ltc_arima_model()

                sell_price = st.number_input("Enter the sell price:", min_value=0.0)
                coin_quantity = st.number_input("Enter the quantity of coins:", min_value=0.0)

                if st.button("Calculate"):
                    # Prepare input data for the ARIMA model (similar to other models)
                    today = datetime.date.today()
                    start_date = datetime.date(2017, 1, 1)



                    coin_data = fetch_historical_price_data(selected_coin, start_date, today)

                    # Use the loaded ARIMA model to make predictions
                    predicted_price = arima_model.predict(n_periods=1)

                    # Calculate potential profit/loss
                    potential_profit = (sell_price - predicted_price[-1]) * coin_quantity

                    # Display the investment scenario results to the user
                    st.write(f"Predicted {selected_coin} Price: {predicted_price[-1]:.2f}")
                    st.write(f"Potential Profit/Loss: ${potential_profit:.2f}")
                    if potential_profit > 0:
                        st.write("Recommendation: Buy Recommendation")
                    else:
                        st.write("Recommendation: Sell Recommendation")

if __name__ == "__main__":
    main()
