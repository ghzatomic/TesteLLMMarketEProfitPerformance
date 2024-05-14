import yfinance as yf
import pandas as pd
import ta

# Função para obter dados históricos
def get_historical_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, interval='1d')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# Função para aplicar indicadores técnicos
def apply_technical_indicators(df):
    df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    return df

# Função para executar a estratégia de trading
def backtest_strategy(df):
    buy_signals = []
    sell_signals = []
    position = None  # 'Long' or 'Short'

    for i in range(len(df)):
        if df['RSI'].iloc[i] < 30 and position != 'Long':
            buy_signals.append((df.index[i], df['Close'].iloc[i]))
            position = 'Long'
        elif df['RSI'].iloc[i] > 70 and position == 'Long':
            sell_signals.append((df.index[i], df['Close'].iloc[i]))
            position = None
    
    return buy_signals, sell_signals

# Função para calcular o desempenho da estratégia
def calculate_performance(buy_signals, sell_signals):
    profits = []
    for buy, sell in zip(buy_signals, sell_signals):
        profit = sell[1] - buy[1]
        profits.append(profit)
    return profits

# Parâmetros do backtesting
symbol = 'PETR4.SA'
start_date = '2020-01-01'
end_date = '2023-01-01'

# Obter dados históricos
df = get_historical_data(symbol, start=start_date, end=end_date)
df = apply_technical_indicators(df)

# Executar a estratégia de backtesting
buy_signals, sell_signals = backtest_strategy(df)

# Calcular desempenho
profits = calculate_performance(buy_signals, sell_signals)
total_profit = sum(profits)
num_trades = len(profits)
average_profit = total_profit / num_trades if num_trades > 0 else 0
accuracy = sum(1 for p in profits if p > 0) / num_trades if num_trades > 0 else 0

# Exibir resultados
print(f"Total Profit: {total_profit}")
print(f"Number of Trades: {num_trades}")
print(f"Average Profit per Trade: {average_profit}")
print(f"Accuracy: {accuracy * 100}%")

# Exibir sinais de compra e venda
print("Buy Signals:", buy_signals)
print("Sell Signals:", sell_signals)
