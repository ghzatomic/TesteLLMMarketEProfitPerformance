import yfinance as yf
import pandas as pd
import numpy as np
import ta
from cvxopt import matrix, solvers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from alpha_vantage.timeseries import TimeSeries
import itertools
import random

# Função para obter dados históricos usando yfinance
def get_historical_data_yfinance(symbols, start, end):
    data = yf.download(symbols, start=start, end=end, interval='1d', group_by='ticker')
    return data

# Função para obter dados históricos usando Alpha Vantage
def get_historical_data_alphavantage(symbols, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data = {}
    for symbol in symbols:
        data[symbol], _ = ts.get_daily(symbol=symbol, outputsize='full')
        data[symbol] = data[symbol]['4. close'].rename('Close')
    data = pd.concat(data.values(), keys=data.keys(), names=['Ticker', 'Date'])
    return data

# Função para aplicar indicadores técnicos
def apply_technical_indicators(df):
    df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'])
    return df

# Função para preparar os dados para a rede neural
def prepare_data_for_nn(df):
    df['Price_Change'] = df['Close'].pct_change().shift(-1)
    df = df.dropna()

    if df.empty:
        raise ValueError("Not enough data to prepare for neural network.")

    X = df[['MA20', 'RSI', 'Bollinger_High', 'Bollinger_Low']]
    y = (df['Price_Change'] > 0).astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test, scaler

# Função para criar e treinar a rede neural
def create_and_train_nn(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)  # Alterado para verbose=2

    return model

# Função para fazer previsões usando a rede neural
def predict_with_nn(model, df, scaler):
    X = df[['MA20', 'RSI', 'Bollinger_High', 'Bollinger_Low']]
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    predictions = model.predict(X_scaled ,verbose = 0)
    df['Prediction'] = (predictions > 0.5).astype(int)
    return df

# Função para executar a estratégia de trading com base nas previsões da rede neural
def backtest_strategy_nn(df):
    buy_signals = []
    sell_signals = []
    position = None

    for i in range(1, len(df)):
        if df['Prediction'].iloc[i] == 1 and position != 'Long':
            buy_signals.append((df.index[i], df['Close'].iloc[i], 100))
            position = 'Long'
        elif df['Prediction'].iloc[i] == 0 and position == 'Long':
            sell_signals.append((df.index[i], df['Close'].iloc[i], 100))
            position = None

    return buy_signals, sell_signals

# Função para executar a estratégia de trading sem usar IA
def backtest_strategy_no_nn(df):
    buy_signals = []
    sell_signals = []
    position = None

    for i in range(1, len(df)):
        if df['RSI'].iloc[i] < 30 and position != 'Long':
            buy_signals.append((df.index[i], df['Close'].iloc[i], 100))
            position = 'Long'
        elif df['RSI'].iloc[i] > 70 and position == 'Long':
            sell_signals.append((df.index[i], df['Close'].iloc[i], 100))
            position = None

    return buy_signals, sell_signals

# Função para calcular o desempenho da estratégia e atualizar o saldo
def calculate_performance(buy_signals, sell_signals, initial_balance):
    balance = initial_balance
    profits = []
    for buy, sell in zip(buy_signals, sell_signals):
        profit = (sell[1] - buy[1]) * buy[2]
        balance += profit
        profits.append(profit)
    return profits, balance

# Função para calcular retornos e matriz de covariância
def calculate_returns_and_covariance(df):
    returns = df.pct_change().dropna()
    mean_returns = returns.mean()
    covariance_matrix = returns.cov()
    return mean_returns, covariance_matrix

# Função para otimizar o portfólio
def optimize_portfolio(mean_returns, covariance_matrix):
    n = len(mean_returns)
    P = matrix(covariance_matrix.values)
    q = matrix(np.zeros(n))
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    
    try:
        sol = solvers.qp(P, q, G, h, A, b)
        weights = np.array(sol['x']).flatten()
        return weights
    except ValueError as e:
        print("Optimization error:", e)
        return None

# Função para gerar o relatório detalhado
def generate_report(transactions):
    report = pd.DataFrame(columns=['Symbol', 'Date', 'Action', 'Price', 'Quantity', 'Profit'])
    for transaction in transactions:
        report = report.append(transaction, ignore_index=True)
    return report

# Função para decidir se deve comprar ou não
def decision_today(df, use_ai):
    # Aplicar indicadores técnicos
    df = apply_technical_indicators(df)

    if use_ai:
        # Preparar dados para a rede neural
        X_train, X_test, y_train, y_test, scaler = prepare_data_for_nn(df)
        if X_train.shape[0] == 0:
            return 0  # Não há dados suficientes para tomar uma decisão

        # Criar e treinar a rede neural
        model = create_and_train_nn(X_train, y_train)

        # Fazer previsões com a rede neural
        df = predict_with_nn(model, df, scaler)

        # Última previsão
        last_prediction = df['Prediction'].iloc[-1]
    else:
        # Último valor do RSI
        last_rsi = df['RSI'].iloc[-1]

        # Decisão baseada no RSI
        last_prediction = 1 if last_rsi < 30 else 0

    return last_prediction

# Função para selecionar as 10 melhores carteiras
def select_top_portfolios(close_prices, num_portfolios=10):
    all_combinations = list(itertools.combinations(close_prices.columns, 5))
    random.shuffle(all_combinations)  # Shuffle to sample randomly
    portfolios = []

    for idx, combination in enumerate(all_combinations[:1000]):  # Limit to 1000 combinations
        print(f"Processing combination {idx+1}/{min(1000, len(all_combinations))}")
        comb_prices = close_prices[list(combination)]
        mean_returns, covariance_matrix = calculate_returns_and_covariance(comb_prices)
        
        weights = optimize_portfolio(mean_returns, covariance_matrix)
        if weights is not None:
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_std_dev
            portfolios.append((combination, portfolio_return, sharpe_ratio))

    portfolios = sorted(portfolios, key=lambda x: x[2], reverse=True)[:num_portfolios]
    return [p[0] for p in portfolios]

# Carregar a lista de ações brasileiras de um arquivo CSV
acoes_brasileiras = pd.read_csv('acoes_brasileiras.csv')['symbol'].tolist()

# Parâmetros do backtesting
start_date = '2024-01-13'
end_date = '2024-03-13'
initial_balance = 1000  # Saldo inicial em reais
run_backtesting = True  # Variável para ativar/desativar o backtesting
use_ai = True  # Variável para ativar/desativar o uso de IA
use_yfinance = True  # Variável para escolher entre yfinance e Alpha Vantage
alpha_vantage_api_key = 'your_alpha_vantage_api_key'  # Substitua pelo seu API key

# Obter dados históricos
if use_yfinance:
    get_historical_data = get_historical_data_yfinance
else:
    get_historical_data = lambda symbols, start, end: get_historical_data_alphavantage(symbols, alpha_vantage_api_key)

# Função para obter e preparar dados históricos
def prepare_historical_data(acoes_brasileiras, start_date, end_date):
    df = get_historical_data(acoes_brasileiras, start_date, end_date)
    close_prices = df.xs('Close', level=1, axis=1)
    close_prices = close_prices.ffill().bfill().dropna(axis=1)
    return close_prices

# Preparar dados históricos
close_prices = prepare_historical_data(acoes_brasileiras, start_date, end_date)

if run_backtesting:
    # Selecionar as 10 melhores carteiras
    selected_symbols_list = select_top_portfolios(close_prices)
else:
    # Selecionar a melhor carteira
    selected_symbols_list = select_top_portfolios(close_prices, num_portfolios=1)

# Coletar dados e aplicar análise técnica para as ações selecionadas
all_profits = []
final_balance = initial_balance
transactions = []

if run_backtesting:
    for selected_symbols in selected_symbols_list:
        daily_balance = initial_balance
        positions = {symbol: 0 for symbol in selected_symbols}
        df_combined = pd.DataFrame()

        for symbol in selected_symbols:
            df_symbol = get_historical_data([symbol], start_date, end_date)
            df_symbol = apply_technical_indicators(df_symbol)
            df_combined[symbol] = df_symbol['Close']

        for current_date in df_combined.index:
            for symbol in selected_symbols:
                df_symbol = df_combined.loc[:current_date, symbol].to_frame().rename(columns={symbol: 'Close'})
                df_symbol = apply_technical_indicators(df_symbol)

                if use_ai:
                    try:
                        X_train, X_test, y_train, y_test, scaler = prepare_data_for_nn(df_symbol)
                    except ValueError:
                        continue  # Se não houver dados suficientes, pule este dia

                    model = create_and_train_nn(X_train, y_train)
                    df_symbol = predict_with_nn(model, df_symbol, scaler)
                    decision = df_symbol['Prediction'].iloc[-1]
                else:
                    decision = decision_today(df_symbol, use_ai)

                if decision == 1 and positions[symbol] == 0:  # Buy
                    quantity = daily_balance // df_symbol['Close'].iloc[-1]
                    buy_price = df_symbol['Close'].iloc[-1]
                    positions[symbol] = quantity
                    daily_balance -= quantity * buy_price
                    transactions.append({
                        'Symbol': symbol,
                        'Date': current_date,
                        'Action': 'Buy',
                        'Price': buy_price,
                        'Quantity': quantity,
                        'Profit': 0
                    })
                elif decision == 0 and positions[symbol] > 0:  # Sell
                    sell_price = df_symbol['Close'].iloc[-1]
                    quantity = positions[symbol]
                    profit = (sell_price - buy_price) * quantity
                    daily_balance += quantity * sell_price
                    positions[symbol] = 0
                    transactions.append({
                        'Symbol': symbol,
                        'Date': current_date,
                        'Action': 'Sell',
                        'Price': sell_price,
                        'Quantity': quantity,
                        'Profit': profit
                    })

        # Atualizar saldo final
        for symbol in selected_symbols:
            final_balance += positions[symbol] * df_combined[symbol].iloc[-1]

    # Gerar relatório detalhado
    report = generate_report(transactions)

    # Exibir os resultados
    print("\nRelatório Detalhado de Ganhos/Perdas Diários:")
    print(report)

    # Exibir saldo final
    print(f"\nSaldo final se tivesse investido R$1000: R${final_balance:.2f}")
else:
    for selected_symbols in selected_symbols_list:
        for symbol in selected_symbols:
            df_symbol = get_historical_data([symbol], start_date, end_date)
            decision = decision_today(df_symbol, use_ai)
            if decision == 1:
                print(f"Hoje você compra {symbol}")
            else:
                print(f"Hoje você não compra {symbol}")
