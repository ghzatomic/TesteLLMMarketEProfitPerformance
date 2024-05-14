import yfinance as yf
import pandas as pd
import numpy as np
import ta
from cvxopt import matrix, solvers

# Função para obter dados históricos
def get_historical_data(symbols, start, end):
    data = yf.download(symbols, start=start, end=end, interval='1d', group_by='ticker')
    return data

# Função para aplicar indicadores técnicos
def apply_technical_indicators(df):
    df.loc[:, 'MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df.loc[:, 'RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df.loc[:, 'Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'])
    df.loc[:, 'Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'])
    return df

# Função para executar a estratégia de trading
def backtest_strategy(df):
    buy_signals = []
    sell_signals = []
    position = None  # 'Long' or 'Short'

    for i in range(1, len(df)):
        if df['RSI'].iloc[i] < 30 and position != 'Long':
            buy_signals.append((df.index[i], df['Close'].iloc[i], 100))  # Compre 100 ações
            position = 'Long'
        elif df['RSI'].iloc[i] > 70 and position == 'Long':
            sell_signals.append((df.index[i], df['Close'].iloc[i], 100))  # Venda 100 ações
            position = None
    
    return buy_signals, sell_signals

# Função para calcular o desempenho da estratégia e atualizar o saldo
def calculate_performance(buy_signals, sell_signals, initial_balance):
    balance = initial_balance
    profits = []
    for buy, sell in zip(buy_signals, sell_signals):
        profit = (sell[1] - buy[1]) * buy[2]  # Preço de venda - preço de compra * quantidade
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
    sol = solvers.qp(P, q, G, h, A, b)
    weights = np.array(sol['x']).flatten()
    return weights

# Função para gerar o relatório detalhado
def generate_report(buy_signals, sell_signals):
    report = pd.DataFrame(columns=['Symbol', 'Date', 'Action', 'Price', 'Quantity', 'Profit'])
    for buy, sell in zip(buy_signals, sell_signals):
        buy_entry = pd.DataFrame({'Symbol': [buy[0]], 'Date': [buy[1]], 'Action': ['Buy'], 'Price': [buy[2]], 'Quantity': [buy[3]], 'Profit': [0]})
        sell_entry = pd.DataFrame({'Symbol': [sell[0]], 'Date': [sell[1]], 'Action': ['Sell'], 'Price': [sell[2]], 'Quantity': [sell[3]], 'Profit': [(sell[2] - buy[2]) * buy[3]]})
        report = pd.concat([report, buy_entry, sell_entry], ignore_index=True)
    return report

# Carregar a lista de ações brasileiras de um arquivo CSV
acoes_brasileiras = pd.read_csv('acoes_brasileiras.csv')['symbol'].tolist()

# Parâmetros do backtesting
start_date = '2020-01-01'
end_date = '2023-01-01'
initial_balance = 1000  # Saldo inicial em reais
run_backtesting = False  # Variável para ativar/desativar o backtesting

if run_backtesting:
    # Obter dados históricos
    df = get_historical_data(acoes_brasileiras, start=start_date, end=end_date)

    # Verificar a estrutura dos dados
    print(df.head())
    print(df.columns)

    # Calcular retornos esperados e matriz de covariância
    close_prices = df.xs('Close', level=1, axis=1)

    # Preencher valores NaNs
    close_prices = close_prices.ffill().bfill()

    # Remover colunas que ainda contêm NaNs após o preenchimento
    close_prices = close_prices.dropna(axis=1)

    mean_returns, covariance_matrix = calculate_returns_and_covariance(close_prices)

    # Verificar se há valores NaNs ou infinitos na matriz de covariância
    if np.any(np.isnan(covariance_matrix)) or np.any(np.isinf(covariance_matrix)):
        raise ValueError("A matriz de covariância contém valores NaNs ou infinitos.")

    # Otimizar o portfólio
    weights = optimize_portfolio(mean_returns, covariance_matrix)

    # Selecionar as melhores ações com base nos pesos
    selected_symbols = [close_prices.columns[i] for i in range(len(close_prices.columns)) if weights[i] > 0.1]
else:
    # Lista de ações pré-selecionadas (ajuste conforme necessário)
    selected_symbols = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA']

# Coletar dados e aplicar análise técnica para as ações selecionadas
all_profits = []
buy_signals_report = []
sell_signals_report = []
final_balance = initial_balance

for symbol in selected_symbols:
    df_symbol = get_historical_data([symbol], start=start_date, end=end_date)

    # Aplicar indicadores técnicos
    df_symbol = apply_technical_indicators(df_symbol)

    # Executar a estratégia de backtesting
    buy_signals, sell_signals = backtest_strategy(df_symbol)

    # Calcular desempenho e atualizar saldo final
    profits, final_balance = calculate_performance(buy_signals, sell_signals, final_balance)
    total_profit = sum(profits)
    num_trades = len(profits)
    average_profit = total_profit / num_trades if num_trades > 0 else 0
    accuracy = sum(1 for p in profits if p > 0) / num_trades if num_trades > 0 else 0

    all_profits.append((symbol, total_profit, num_trades, average_profit, accuracy))
    
    buy_signals_report.extend([(symbol, bs[0], bs[1], bs[2]) for bs in buy_signals])
    sell_signals_report.extend([(symbol, ss[0], ss[1], ss[2]) for ss in sell_signals])

# Classificar ações por lucro total
all_profits = sorted(all_profits, key=lambda x: x[1], reverse=True)[:10]

# Gerar relatório detalhado
report = generate_report(buy_signals_report, sell_signals_report)

# Exibir os resultados
print("As 10 melhores ações para day trade baseadas em dados históricos são:")
for stock in all_profits:
    print(f"Ação: {stock[0]}, Lucro Total: {stock[1]:.2f}, Número de Trades: {stock[2]}, Lucro Médio por Trade: {stock[3]:.2f}, Precisão: {stock[4] * 100:.2f}%")

# Exibir relatório detalhado
print("\nRelatório Detalhado de Ganhos/Perdas Diários:")
print(report)

# Exibir saldo final
print(f"\nSaldo final se tivesse investido R$1000: R${final_balance:.2f}")
