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
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'])
    return df

# Função para executar a estratégia de trading
def backtest_strategy(df):
    buy_signals = []
    sell_signals = []
    position = None  # 'Long' or 'Short'

    for i in range(1, len(df)):
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
symbols = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA']
start_date = '2020-01-01'
end_date = '2023-01-01'

# Coletar dados e aplicar análise técnica
all_profits = []
for symbol in symbols:
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

    all_profits.append((symbol, total_profit, num_trades, average_profit, accuracy))

# Classificar ações por lucro total
all_profits = sorted(all_profits, key=lambda x: x[1], reverse=True)[:10]

# Exibir os resultados
print("As 10 melhores ações para day trade baseadas em dados históricos são:")
for stock in all_profits:
    print(f"Ação: {stock[0]}, Lucro Total: {stock[1]:.2f}, Número de Trades: {stock[2]}, Lucro Médio por Trade: {stock[3]:.2f}, Precisão: {stock[4] * 100:.2f}%")
