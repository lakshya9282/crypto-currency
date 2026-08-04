# live_crypto_dashboard_full.py
# ================================================
# LIVE MULTI-CRYPTO DASHBOARD WITH ML PREDICTIONS (BINANCE EDITION)
# - Volume-based features
# - Top 5 gainers / losers (24h)
# - Download buttons for predictions & raw data
# ================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import time

st.set_page_config(page_title="Live Crypto Dashboard", layout="wide")
st.title("Live Multi-Crypto Dashboard with ML Predictions")
st.markdown("Real-time cryptocurrency prices with next-day trend & price predictions. Powered by **Binance API** for fast, reliable data.")

# ---- AUTO REFRESH EVERY HOUR ----
st_autorefresh(interval=3600000, key="crypto_refresh")

# ---- SYMBOL DICTIONARY (Binance Tickers) ----
coins_dict = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Dogecoin (DOGE)": "DOGEUSDT",
    "Solana (SOL)": "SOLUSDT",
    "Cardano (ADA)": "ADAUSDT",
    "Litecoin (LTC)": "LTCUSDT",
    "Binance Coin (BNB)": "BNBUSDT",
    "Ripple (XRP)": "XRPUSDT",
    "Polkadot (DOT)": "DOTUSDT",
    "Avalanche (AVAX)": "AVAXUSDT",
    "Shiba Inu (SHIB)": "SHIBUSDT",
    "Tron (TRX)": "TRXUSDT",
    "Chainlink (LINK)": "LINKUSDT",
    "Polygon (MATIC)": "MATICUSDT",
    "Stellar (XLM)": "XLMUSDT"
}

reverse_coins_dict = {v: k for k, v in coins_dict.items()}

# ---- SESSION STATE SETUP ----
if "selected_coins" not in st.session_state:
    st.session_state.selected_coins = ["Bitcoin (BTC)", "Ethereum (ETH)"]

# ---- TOP GAINERS / LOSERS SECTION (BINANCE) ----
st.markdown("##  Top 5 Gainers & Top 5 Losers (24h)")

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_binance_markets():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Filter only for the USDT pairs we track in our dictionary to avoid obscure coins
        tracked_symbols = list(coins_dict.values())
        filtered_data = [item for item in data if item['symbol'] in tracked_symbols]
        
        df = pd.DataFrame(filtered_data)
        df["priceChangePercent"] = pd.to_numeric(df["priceChangePercent"])
        df["lastPrice"] = pd.to_numeric(df["lastPrice"])
        df["volume"] = pd.to_numeric(df["volume"])
        return df
    except Exception as e:
        st.error(f"Error fetching Binance markets: {e}")
        return pd.DataFrame()

df_markets = fetch_binance_markets()

if not df_markets.empty:
    sorted_by_change = df_markets.sort_values(by="priceChangePercent", ascending=False)
    top_gainers = sorted_by_change.head(5)
    top_losers = sorted_by_change.tail(5).sort_values(by="priceChangePercent")
else:
    top_gainers = pd.DataFrame()
    top_losers = pd.DataFrame()

def format_market_table(df):
    if df.empty: return df
    display_df = df[["symbol", "lastPrice", "priceChangePercent", "volume"]].copy()
    display_df["Name"] = display_df["symbol"].map(reverse_coins_dict)
    display_df = display_df[["Name", "symbol", "lastPrice", "priceChangePercent", "volume"]]
    display_df.rename(columns={
        "symbol": "Symbol", 
        "lastPrice": "Price (USD)",
        "priceChangePercent": "24h %", 
        "volume": "24h Volume"
    }, inplace=True)
    return display_df.reset_index(drop=True)

col1, col2, col3 = st.columns([3, 1, 3])
with col1:
    st.subheader("Top 5 Gainers (24h)")
    if not top_gainers.empty:
        st.table(format_market_table(top_gainers))
    else:
        st.info("Market data unavailable.")

with col2:
    st.write("") 
    if st.button(" Add Gainers"):
        st.session_state.selected_coins = [reverse_coins_dict[sym] for sym in top_gainers["symbol"].tolist()]
        st.experimental_rerun()
    if st.button(" Add Losers"):
        st.session_state.selected_coins = [reverse_coins_dict[sym] for sym in top_losers["symbol"].tolist()]
        st.experimental_rerun()

with col3:
    st.subheader("Top 5 Losers (24h)")
    if not top_losers.empty:
        st.table(format_market_table(top_losers))
    else:
        st.info("Market data unavailable.")

# ---- USER INPUT: MULTISELECT ----
st.markdown("---")
selected_coins = st.multiselect(
    "Select Cryptocurrencies:",
    list(coins_dict.keys()),
    default=st.session_state.selected_coins
)
timeframe = st.selectbox("Select Timeframe (days):", [7, 30, 90, 180, 365, 1000], index=3)

if not selected_coins:
    st.warning("Please select at least one cryptocurrency to proceed.")
    st.stop()

# ---- FETCH HISTORICAL DATA (BINANCE KLINES) ----
@st.cache_data(ttl=3600)
def fetch_binance_history(symbol: str, days: int):
    # Binance klines endpoint
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": days
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Binance kline format: [Open time, Open, High, Low, Close, Volume, Close time, ...]
        if not data:
            return pd.DataFrame(columns=["Date", "Price", "Volume"])
            
        df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Price", "Volume", "CloseTime", "QAV", "NumTrades", "TBAV", "TQAV", "Ignore"])
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df["Price"] = pd.to_numeric(df["Price"])
        df["Volume"] = pd.to_numeric(df["Volume"])
        
        return df[["Date", "Price", "Volume"]].reset_index(drop=True)
    except Exception as e:
        st.error(f"Failed to fetch history for {symbol}: {e}")
        return pd.DataFrame(columns=["Date", "Price", "Volume"])

st.info("Fetching live historical data from Binance...")
coin_data_dict = {}

# No more aggressive progress bar delays needed! Binance is fast.
for coin_name in selected_coins:
    symbol = coins_dict.get(coin_name)
    if symbol:
        coin_data_dict[coin_name] = fetch_binance_history(symbol, timeframe)

st.success("Data fetching complete!")

# ---- FEATURE ENGINEERING + ML (Unchanged logic) ----
st.markdown("##  Predictions & Signals (with Volume features)")
results = []
all_raw_for_download = {}
all_predictions_for_download = []

for coin_name, df in coin_data_dict.items():
    raw_df = df.copy()
    all_raw_for_download[coin_name] = raw_df

    if df.empty or len(df) < 5:
        results.append({
            "Coin": coin_name, "Predicted Price": "Insufficient data", 
            "Predicted Trend": "N/A", "Signal": "N/A", "MAE": "N/A", "Accuracy": "N/A"
        })
        continue

    # Feature engineering
    df = df.copy().reset_index(drop=True)
    df["MA7"] = df["Price"].rolling(window=7, min_periods=1).mean()
    df["MA14"] = df["Price"].rolling(window=14, min_periods=1).mean()
    df["Momentum"] = df["Price"] - df["Price"].shift(1)
    df["Return"] = df["Price"].pct_change().fillna(0)
    df["Volatility"] = df["Return"].rolling(window=7, min_periods=1).std().fillna(0)
    df["Volume_Change"] = df["Volume"].pct_change().fillna(0)
    df["Volume_Ratio"] = df["Volume"] / (df["Volume"].rolling(window=7, min_periods=1).mean().replace(0, np.nan)).fillna(0)
    df["Trend"] = (df["Price"].shift(-1) > df["Price"]).astype(int)
    df = df.dropna().reset_index(drop=True)

    features = ["Price", "MA7", "MA14", "Momentum", "Return", "Volatility", "Volume", "Volume_Change", "Volume_Ratio"]
    X = df[features].iloc[:-1].reset_index(drop=True)
    y_price = df["Price"].shift(-1).iloc[:-1].reset_index(drop=True)
    y_trend = df["Trend"].iloc[:-1].reset_index(drop=True)

    if len(X) < 5:
        continue

    test_size = 0.2
    split_idx = max(1, int(len(X) * (1 - test_size)))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_price, y_test_price = y_price.iloc[:split_idx], y_price.iloc[split_idx:]
    y_train_trend, y_test_trend = y_trend.iloc[:split_idx], y_trend.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg = LinearRegression()
    cls = LogisticRegression(solver="liblinear", max_iter=1000)

    try:
        reg.fit(X_train_scaled, y_train_price)
        cls.fit(X_train_scaled, y_train_trend)
        y_pred_price = reg.predict(X_test_scaled)
        y_pred_trend = cls.predict(X_test_scaled)
        mae = np.mean(np.abs(y_test_price.values - y_pred_price))
        acc = np.mean(y_test_trend.values == y_pred_trend)
    except Exception:
        mae, acc = np.nan, np.nan

    try:
        latest_scaled = X_test_scaled[-1].reshape(1, -1)
    except:
        latest_scaled = X_train_scaled[-1].reshape(1, -1)

    try:
        pred_price = float(reg.predict(latest_scaled)[0])
        pred_trend = int(cls.predict(latest_scaled)[0])
        signal = "Buy" if pred_trend == 1 else "Sell"
        pred_trend_str = "UP" if pred_trend == 1 else "DOWN"
        pred_price_rounded = round(pred_price, 4)
    except:
        pred_price_rounded, pred_trend_str, signal = "N/A", "N/A", "N/A"

    results.append({
        "Coin": coin_name, "Predicted Price": pred_price_rounded, "Predicted Trend": pred_trend_str,
        "Signal": signal, "MAE": round(mae, 4) if not np.isnan(mae) else "N/A",
        "Accuracy": f"{acc*100:.2f}%" if not np.isnan(acc) else "N/A"
    })

    all_predictions_for_download.append({
        "Coin": coin_name, "Predicted_Price": pred_price_rounded, "Predicted_Trend": pred_trend_str,
        "Signal": signal, "MAE": round(mae, 4) if not np.isnan(mae) else None,
        "Accuracy": acc if not np.isnan(acc) else None, "Generated_At": datetime.utcnow().isoformat() + "Z"
    })

st.dataframe(pd.DataFrame(results))

# ---- MULTI-COIN INTERACTIVE CHART ----
st.markdown("## 📈 Price Chart")
fig = go.Figure()
for coin_name in selected_coins:
    df = coin_data_dict.get(coin_name, pd.DataFrame())
    if not df.empty:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Price"], mode="lines", name=coin_name))
fig.update_layout(title="Cryptocurrency Prices", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# ---- BUY/SELL SUMMARY ----
st.markdown("## 💹 Buy/Sell Signals")
for res in results:
    coin, sig, pred, price = res.get("Coin"), res.get("Signal"), res.get("Predicted Trend"), res.get("Predicted Price")
    if sig == "Buy": st.success(f"{coin}: BUY (Predicted Trend: {pred}, Price: ${price})")
    elif sig == "Sell": st.error(f"{coin}: SELL (Predicted Trend: {pred}, Price: ${price})")
