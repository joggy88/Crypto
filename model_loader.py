from tensorflow import keras

import joblib
def load_btc_lstm_model():
    return keras.models.load_model('AE2/mybtc_model.keras')


def load_eth_lstm_model():
    return keras.models.load_model('AE2/myeth_model.keras')

def load_ltc_lstm_model():
    return keras.models.load_model('AE2/myltc_model.keras')



def load_btc_rf_model():
    return joblib.load('AE2/btc_rf_model.joblib')

def load_eth_rf_model():
    return joblib.load('AE2/eth_rf_model.joblib')

def load_ltc_rf_model():
    return joblib.load('AE2/ltc_rf_model.joblib')



def load_btc_arima_model():
    return joblib.load('AE2/btc_arima_model.pkl')

def load_eth_arima_model():
    return joblib.load('AE2/eth_arima_model.pkl')

def load_ltc_arima_model():
    return joblib.load('AE2/ltc_arima_model.pkl')
