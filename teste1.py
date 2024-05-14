import requests
import pandas as pd
import ta

# Configurações da API (exemplo com Alpha Vantage)
API_KEY = 'AIH3L65MS5ZI87GO'  # Substitua pela sua chave de API
BASE_URL = 'https://www.alphavantage.co/query'

# Função para obter dados de ações
def get_stock_data(symbol):
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': '1min',
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return pd.DataFrame(data['Time Series (1min)']).T.astype(float)

# Lista de ações para análise (exemplo com algumas ações da B3)
symbols = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA']

# Coletar dados e aplicar análise técnica
stock_data = {}
for symbol in symbols:
    df = get_stock_data(symbol)
    df['MA'] = ta.trend.sma_indicator(df['4. close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['4. close'], window=14)
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['4. close'])
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['4. close'])
    stock_data[symbol] = df

# Definir critérios de filtragem (exemplo: ações com RSI < 30)
selected_stocks = []
for symbol, df in stock_data.items():
    if df['RSI'].iloc[-1] < 30:
        selected_stocks.append((symbol, df['4. close'].iloc[-1]))

# Classificar ações por fechamento mais baixo (exemplo)
selected_stocks = sorted(selected_stocks, key=lambda x: x[1])[:10]

# Exibir as 10 melhores ações
print("As 10 melhores ações para day trade são:")
for stock in selected_stocks:
    print(stock[0])
