
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

# 1. DownloadingS historical stock data
ticker = "RELIANCE.NS"

data = yf.download(
    ticker,
    period="5y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

# 2. Check whether data was downloaded
if data.empty:
    print("No data downloaded. Check your internet connection.")
    raise SystemExit

# Handle possible multi-level column names
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print("First five rows of downloaded data:")
print(data.head())

print("\nOriginal dataset shape:", data.shape)

print("\nMissing values before cleaning:")
print(data.isna().sum())

# 3. Clean the data
data = data[~data.index.duplicated(keep="first")]
data = data.dropna()

print("\nDataset shape after cleaning:", data.shape)

# 4. Calculate the 20-day Simple Moving Average
data["SMA_20"] = data["Close"].rolling(window=20).mean()
print("\nLatest ten rows:")
print(data[["Close", "SMA_20"]].tail(10))

# 5. Display the latest ten closing prices and moving averages
print("\nLatest ten rows:")
print(data[["Close", "SMA_20"]].tail(10))

# Find days when closing price is above SMA-20
above_average = data[data["Close"] > data["SMA_20"]]

print(above_average[["Close", "SMA_20"]].tail(10))
# Calculate rolling standard deviation
data["STD_20"] = data["Close"].rolling(window=20).std()

# Calculate Z-score
data["Z_Score"] = (
    (data["Close"] - data["SMA_20"]) / data["STD_20"]
)
data["Target_Position"] = 0

position = 0

for i in range(len(data)):
    z = data["Z_Score"].iloc[i]

    if pd.isna(z):
        data.iloc[
            i, data.columns.get_loc("Target_Position")
        ] = position
        continue

    if position == 0 and z <= -2:
        position = 1

    elif position == 1 and z >= 0:
        position = 0

    data.iloc[
        i, data.columns.get_loc("Target_Position")
    ] = position

print(data[["Close", "Z_Score", "Target_Position"]].tail(20))

# Display the latest values
print(data[["Close", "SMA_20", "STD_20", "Z_Score"]].tail(10))


# A signal made at a day's close executes at the next trading day's open.

# Position held from the previous close until today's open
data["Overnight_Position"] = (
    data["Target_Position"].shift(2).fillna(0)
)

# Position held from today's open until today's close
data["Intraday_Position"] = (
    data["Target_Position"].shift(1).fillna(0)
)

# Stock returns during each interval
data["Overnight_Return"] = (
    data["Open"] / data["Close"].shift(1) - 1
)

data["Intraday_Return"] = (
    data["Close"] / data["Open"] - 1
)

# Returns earned by the strategy
data["Overnight_Strategy_Return"] = (
    data["Overnight_Position"] * data["Overnight_Return"]
)

data["Intraday_Strategy_Return"] = (
    data["Intraday_Position"] * data["Intraday_Return"]
)


# Assumed transaction cost: 0.1% per transaction
transaction_cost_rate = 0.001

#  Calculate how much our position changes at today's open
data["Turnover"] = abs(
    data["Intraday_Position"] - data["Overnight_Position"]
)

#  Calculate transaction costs as a fraction of portfolio value
data["Transaction_Cost"] = (
    data["Turnover"] * transaction_cost_rate
)

#  Calculate the strategy return after transaction costs
data["Strategy_Return"] = (
    (1 + data["Overnight_Strategy_Return"])
    * (1 - data["Transaction_Cost"])
    * (1 + data["Intraday_Strategy_Return"])
    - 1
)

# Display the results
print("\nReturns after transaction costs:")
print(data[[
    "Overnight_Position",
    "Intraday_Position",
    "Turnover",
    "Transaction_Cost",
    "Strategy_Return"
]].tail(10))

print("\nCorrected return calculation results:")
print(data[[
    "Open", "Close", "Target_Position",
    "Overnight_Position", "Intraday_Position",
    "Overnight_Return", "Intraday_Return",
    "Strategy_Return"
]].tail(10))


   
# Initial amount of money in the portfolio
initial_capital = 100000

# Calculate the strategy portfolio value over time
strategy_returns = data["Strategy_Return"].fillna(0)

data["Strategy_Equity"] = (
    initial_capital * (1 + strategy_returns).cumprod()
)

#  Calculate buy-and-hold portfolio value
data["Buy_Hold_Return"] = (
    data["Close"].pct_change().fillna(0)
)

data["Buy_Hold_Equity"] = (
    initial_capital * (1 + data["Buy_Hold_Return"]).cumprod()
)

# Calculate total returns
strategy_total_return = (
    data["Strategy_Equity"].iloc[-1] / initial_capital - 1
)

buy_hold_total_return = (
    data["Buy_Hold_Equity"].iloc[-1] / initial_capital - 1
)

print("\nFinal portfolio values:")
print("Strategy:", round(data["Strategy_Equity"].iloc[-1], 2))
print("Buy and hold:", round(data["Buy_Hold_Equity"].iloc[-1], 2))

print("\n--- Performance Summary ---")
print(f"Strategy total return: {strategy_total_return:.2%}")
print(f"Buy-and-hold total return: {buy_hold_total_return:.2%}")

# Calculate annualized compounded returns
periods = data["Strategy_Return"].notna().sum()

if periods > 0:
    strategy_annual_return = (
        (data["Strategy_Equity"].iloc[-1] / initial_capital)
        ** (252 / periods) - 1
    )

    buy_hold_annual_return = (
        (data["Buy_Hold_Equity"].iloc[-1] / initial_capital)
        ** (252 / periods) - 1
    )

    print(f"Strategy annualized return: {strategy_annual_return:.2%}")
    print(f"Buy-and-hold annualized return: {buy_hold_annual_return:.2%}")

# Highest portfolio value reached up to each date
data["Portfolio_Peak"] = (
    data["Strategy_Equity"].cummax()
)

# Percentage fall from the previous peak
data["Drawdown"] = (
    data["Strategy_Equity"] / data["Portfolio_Peak"] - 1
)

# Most severe drawdown
max_drawdown = data["Drawdown"].min()

print(f"Maximum drawdown: {max_drawdown:.2%}")

import numpy as np

# Daily strategy returns
returns = data["Strategy_Return"].dropna()

# Annualization convention
trading_days = 252

# Annualized volatility
annual_volatility = (
    returns.std() * np.sqrt(trading_days)
)

# Annualized average return
annual_mean_return = (
    returns.mean() * trading_days
)

# Sharpe ratio: assume risk-free rate = 0%
if returns.std() != 0:
    sharpe_ratio = (
        returns.mean() / returns.std()
    ) * np.sqrt(trading_days)
else:
    sharpe_ratio = np.nan

# Sortino ratio: downside deviation
downside_returns = np.minimum(returns, 0)
downside_deviation = np.sqrt(
    np.mean(downside_returns ** 2)
)

if downside_deviation != 0:
    sortino_ratio = (
        returns.mean() / downside_deviation
    ) * np.sqrt(trading_days)
else:
    sortino_ratio = np.nan

print("\n--- Risk Metrics ---")
print(f"Annualized volatility: {annual_volatility:.2%}")
print(f"Sharpe ratio: {sharpe_ratio:.3f}")
print(f"Sortino ratio: {sortino_ratio:.3f}")


# Split the data chronologically: 80% development, 20% testing
split_index = int(len(data) * 0.80)

development_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

print("\n--- Development and Test Periods ---")
print("Development start:", development_data.index.min())
print("Development end:", development_data.index.max())
print("Development rows:", len(development_data))

print("Test start:", test_data.index.min())
print("Test end:", test_data.index.max())
print("Test rows:", len(test_data))


# Recalculate positions for the test period, starting flat
test_data["Test_Target_Position"] = 0
position = 0

for i in range(len(test_data)):
    z = test_data["Z_Score"].iloc[i]

    if pd.isna(z):
        test_data.iloc[
            i, test_data.columns.get_loc("Test_Target_Position")
        ] = position
        continue

    if position == 0 and z <= -2:
        position = 1
    elif position == 1 and z >= 0:
        position = 0

    test_data.iloc[
        i, test_data.columns.get_loc("Test_Target_Position")
    ] = position


# Determine positions for each return interval
test_data["Overnight_Position"] = (
    test_data["Test_Target_Position"].shift(2).fillna(0)
)
test_data["Intraday_Position"] = (
    test_data["Test_Target_Position"].shift(1).fillna(0)
)

# Calculate overnight and intraday market returns
test_data["Overnight_Return"] = (
    test_data["Open"] / test_data["Close"].shift(1) - 1
).fillna(0)

test_data["Intraday_Return"] = (
    test_data["Close"] / test_data["Open"] - 1
).fillna(0)

# Apply positions to the returns
test_data["Overnight_Strategy_Return"] = (
    test_data["Overnight_Position"] * test_data["Overnight_Return"]
)
test_data["Intraday_Strategy_Return"] = (
    test_data["Intraday_Position"] * test_data["Intraday_Return"]
)

# Apply transaction costs when the position changes
test_data["Turnover"] = abs(
    test_data["Intraday_Position"] - test_data["Overnight_Position"]
)
test_data["Transaction_Cost"] = (
    test_data["Turnover"] * transaction_cost_rate
)

# Combine returns and costs
test_data["Strategy_Return"] = (
    (1 + test_data["Overnight_Strategy_Return"])
    * (1 - test_data["Transaction_Cost"])
    * (1 + test_data["Intraday_Strategy_Return"])
    - 1
)

# Build test-period equity curves
initial_capital = 100000

test_data["Strategy_Equity"] = (
    initial_capital * (1 + test_data["Strategy_Return"]).cumprod()
)

test_data["Buy_Hold_Return"] = (
    test_data["Close"].pct_change().fillna(0)
)
test_data["Buy_Hold_Equity"] = (
    initial_capital * (1 + test_data["Buy_Hold_Return"]).cumprod()
)

# Calculate total and annualized returns
periods = len(test_data)

strategy_total_return = (
    test_data["Strategy_Equity"].iloc[-1] / initial_capital - 1
)
buy_hold_total_return = (
    test_data["Buy_Hold_Equity"].iloc[-1] / initial_capital - 1
)

strategy_annual_return = (
    (1 + strategy_total_return) ** (252 / periods) - 1
)
buy_hold_annual_return = (
    (1 + buy_hold_total_return) ** (252 / periods) - 1
)

# Calculate maximum drawdown
test_data["Portfolio_Peak"] = test_data["Strategy_Equity"].cummax()
test_data["Drawdown"] = (
    test_data["Strategy_Equity"] / test_data["Portfolio_Peak"] - 1
)
max_drawdown = test_data["Drawdown"].min()

# Calculate risk metrics
returns = test_data["Strategy_Return"]
daily_std = returns.std()
annual_volatility = daily_std * np.sqrt(252)

if daily_std != 0:
    sharpe_ratio = (returns.mean() / daily_std) * np.sqrt(252)
else:
    sharpe_ratio = np.nan

downside_returns = np.minimum(returns, 0)
downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

if downside_deviation != 0:
    sortino_ratio = (
        returns.mean() / downside_deviation
    ) * np.sqrt(252)
else:
    sortino_ratio = np.nan

# Display the test-period results
print("\n--- OUT-OF-SAMPLE TEST RESULTS ---")
print("Test period:", test_data.index.min().date(),
      "to", test_data.index.max().date())
print("Number of trading observations:", periods)

print(f"Strategy final value: ₹{test_data['Strategy_Equity'].iloc[-1]:,.2f}")
print(f"Buy-and-hold final value: ₹{test_data['Buy_Hold_Equity'].iloc[-1]:,.2f}")

print(f"Strategy total return: {strategy_total_return:.2%}")
print(f"Buy-and-hold total return: {buy_hold_total_return:.2%}")
print(f"Strategy annualized return: {strategy_annual_return:.2%}")
print(f"Buy-and-hold annualized return: {buy_hold_annual_return:.2%}")
print(f"Maximum drawdown: {max_drawdown:.2%}")
print(f"Annualized volatility: {annual_volatility:.2%}")
print(f"Sharpe ratio: {sharpe_ratio:.3f}")
print(f"Sortino ratio: {sortino_ratio:.3f}")


# Buy-and-hold risk metrics for the test period

# 1. Find the highest portfolio value reached so far
test_data["Buy_Hold_Peak"] = (
    test_data["Buy_Hold_Equity"].cummax()
)

# 2. Calculate buy-and-hold drawdown
test_data["Buy_Hold_Drawdown"] = (
    test_data["Buy_Hold_Equity"]
    / test_data["Buy_Hold_Peak"] - 1
)

buy_hold_max_drawdown = test_data["Buy_Hold_Drawdown"].min()

# 3. Calculate annualized volatility
buy_hold_returns = test_data["Buy_Hold_Return"]
buy_hold_std = buy_hold_returns.std()

buy_hold_volatility = buy_hold_std * np.sqrt(252)

# 4. Calculate Sharpe ratio
if buy_hold_std != 0:
    buy_hold_sharpe = (
        buy_hold_returns.mean() / buy_hold_std
    ) * np.sqrt(252)
else:
    buy_hold_sharpe = np.nan

# 5. Calculate Sortino ratio
buy_hold_downside = np.minimum(buy_hold_returns, 0)

buy_hold_downside_deviation = np.sqrt(
    np.mean(buy_hold_downside ** 2)
)

if buy_hold_downside_deviation != 0:
    buy_hold_sortino = (
        buy_hold_returns.mean()
        / buy_hold_downside_deviation
    ) * np.sqrt(252)
else:
    buy_hold_sortino = np.nan

# 6. Display the comparison
print("\n--- RISK COMPARISON: TEST PERIOD ---")
print(f"Strategy maximum drawdown: {max_drawdown:.2%}")
print(f"Buy-and-hold maximum drawdown: {buy_hold_max_drawdown:.2%}")

print(f"Strategy volatility: {annual_volatility:.2%}")
print(f"Buy-and-hold volatility: {buy_hold_volatility:.2%}")

print(f"Strategy Sharpe ratio: {sharpe_ratio:.3f}")
print(f"Buy-and-hold Sharpe ratio: {buy_hold_sharpe:.3f}")

print(f"Strategy Sortino ratio: {sortino_ratio:.3f}")
print(f"Buy-and-hold Sortino ratio: {buy_hold_sortino:.3f}")


# Build a trade log using actual execution timing

trades = []
in_trade = False
entry_date = None
entry_price = None

for i in range(len(test_data)):
    overnight_position = test_data["Overnight_Position"].iloc[i]
    intraday_position = test_data["Intraday_Position"].iloc[i]

    current_date = test_data.index[i]
    current_open = test_data["Open"].iloc[i]

    # Enter when exposure changes from flat to long at the open
    if overnight_position == 0 and intraday_position == 1:
        entry_date = current_date
        entry_price = current_open
        in_trade = True

    # Exit when exposure changes from long to flat at the open
    elif overnight_position == 1 and intraday_position == 0 and in_trade:
        exit_date = current_date
        exit_price = current_open

        gross_return = exit_price / entry_price - 1

        net_return = (
            (exit_price / entry_price)
            * (1 - transaction_cost_rate) ** 2
            - 1
        )

        trades.append({
            "Entry_Date": entry_date,
            "Exit_Date": exit_date,
            "Entry_Price": entry_price,
            "Exit_Price": exit_price,
            "Gross_Return_Percent": gross_return * 100,
            "Net_Return_Percent": net_return * 100
        })

        in_trade = False

# Display the corrected trade log
trade_log = pd.DataFrame(trades)

print("\n--- CORRECTED TRADE LOG ---")

if not trade_log.empty:
    print(trade_log.to_string(index=False))
    print("\nCompleted trades:", len(trade_log))
else:
    print("No completed trades found.")

if in_trade:
    print("\nA trade remains open at the end of the test period.")


# Calculate trade-level performance statistics

if not trade_log.empty:
    returns = trade_log["Net_Return_Percent"] / 100

    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]

    total_trades = len(returns)
    win_rate = len(winning_trades) / total_trades * 100

    average_win = (
        winning_trades.mean() * 100
        if len(winning_trades) > 0 else 0
    )

    average_loss = (
        losing_trades.mean() * 100
        if len(losing_trades) > 0 else 0
    )

    gross_profit = winning_trades.sum()
    gross_loss = abs(losing_trades.sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else np.nan
    )

    compounded_trade_return = (
        (1 + returns).prod() - 1
    ) * 100

    print("\n--- TRADE STATISTICS ---")
    print("Total completed trades:", total_trades)
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average winning trade: {average_win:.2f}%")
    print(f"Average losing trade: {average_loss:.2f}%")
    print(f"Profit factor: {profit_factor:.3f}")
    print(f"Compounded trade return: {compounded_trade_return:.2f}%")

# Create one figure containing three graphs
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Graph 1: Reliance closing price and 20-day moving average
data[["Close", "SMA_20"]].plot(ax=axes[0])
axes[0].set_title("Reliance Closing Price and 20-Day SMA")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Price (INR)")
axes[0].grid(True)

# Graph 2: Strategy drawdown
data["Drawdown"].plot(ax=axes[1])
axes[1].set_title("Strategy Drawdown")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Drawdown")
axes[1].grid(True)

# Graph 3: Strategy versus buy-and-hold
data[["Strategy_Equity", "Buy_Hold_Equity"]].plot(ax=axes[2])
axes[2].set_title("Mean-Reversion Strategy vs Buy and Hold")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Portfolio Value (INR)")
axes[2].grid(True)

# Prevent labels from overlapping
plt.tight_layout()

# Display the complete figure in one window
plt.show()
