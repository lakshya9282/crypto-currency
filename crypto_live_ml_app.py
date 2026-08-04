# live_crypto_dashboard_full.py
# ================================================
# LIVE MULTI-CRYPTO DASHBOARD WITH ML PREDICTIONS (YAHOO FINANCE EDITION)
# - Volume-based features
# - Top gainers / losers (24h) from tracked list
# - Download buttons for predictions & raw data
# ================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

st.set_page_config(page_title="Live Crypto Dashboard", layout="wide")
st.title("Live Multi-Crypto Dashboard with ML Predictions")
st.markdown("Real-time cryptocurrency prices with next-day trend & price predictions. Powered by **Yahoo Finance** for unrestricted, fast data.")

# ---- AUTO REFRESH EVERY HOUR ----
st_autorefresh(interval=3600000, key="crypto_refresh")

# ---- SYMBOL DICTIONARY (Yahoo Finance Tickers) ----
coins_dict = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Ripple (XRP)": "XRP-USD",
    "Polkadot (DOT)": "DOT-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Shiba Inu (SHIB)": "SHIB-USD",
    "Tron (TRX)": "TRX-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Polygon (MATIC)": "MATIC-USD",
    "Stellar (XLM)": "XLM-USD"
}

reverse_coins_dict = {v: k for k, v in coins_dict.items()}

# ---- SESSION STATE SETUP ----
if "selected_coins" not in st.session_state:
    st.session_state.selected_coins = ["Bitcoin (BTC)", "Ethereum (ETH)"]

# ---- TOP GAINERS / LOSERS SECTION (YAHOO FINANCE) ----
st.markdown("##  Top 5 Gainers & Top 5 Losers (24h)")
st.caption("Calculated from your tracked portfolio list below.")

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_market_movers():
    market_data = []
    # Fetch recent data for our tracked coins to find gainers/losers
    for name, ticker in coins_dict.items():
        try:
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period="5d") # Get enough days to ensure we have the last 2 closes
            if len(hist) >= 2:
                prev_close = float(hist['Close'].iloc[-2])
                curr_close = float(hist['Close'].iloc[-1])
                volume = float(hist['Volume'].iloc[-1])
                pct_change = ((curr_close - prev_close) / prev_close) * 100
                
                market_data.append({
                    "name": name,
                    "symbol": ticker,
                    "current_price": curr_close,
                    "price_change_percentage_24h": pct_change,
                    "total_volume": volume
                })
        except Exception:
            continue
    return pd.DataFrame(market_data)

df_markets = fetch_market_movers()

if not df_markets.empty:
    sorted_by_change = df_markets.sort_values(by="price_change_percentage_24h", ascending=False)
    top_gainers = sorted_by_change.head(5)
    top_losers = sorted_by_change.tail(5).sort_values(by="price_change_percentage_24h")
else:
    top_gainers = pd.DataFrame()
    top_losers = pd.DataFrame()

def format_market_table(df):
    if df.empty: return df
    display_df = df[["name", "symbol", "current_price", "price_change_percentage_24h", "total_volume"]].copy()
    display_df.rename(columns={
        "name": "Name",
        "symbol": "Symbol", 
        "current_price": "Price (USD)",
        "price_change_percentage_24h": "24h %", 
        "total_volume": "24h Volume"
    }, inplace=True)
    return display_df.reset_index(drop=True)

col1, col2, col3 = st.columns([3, 1, 3])
with col1:
    st.subheader("Top 5 Gainers")
    if not top_gainers.empty:
        st.table(format_market_table(top_gainers))
    else:
        st.info("Market data unavailable.")

with col2:
    st.write("") 
    if st.button(" Add Gainers"):
        st.session_state.selected_coins = top_gainers["name"].tolist()
        st.experimental_rerun()
    if st.button(" Add Losers"):
        st.session_state.selected_coins = top_losers["name"].tolist()
        st.experimental_rerun()

with col3:
    st.subheader("Top 5 Losers")
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

# ---- FETCH HISTORICAL DATA (YAHOO FINANCE) ----
@st.cache_data(ttl=3600)
def fetch_yf_history(ticker: str, days: int):
    try:
        tkr = yf.Ticker(ticker)
        # Calculate dynamic date range based on user input
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=days)
        
        df = tkr.history(start=start_date, end=end_date)
        
        if df.empty:
            return pd.DataFrame(columns=["Date", "Price", "Volume"])
            
        df = df.reset_index()
        # Handle cases where YF returns 'Datetime' vs 'Date'
        date_col = 'Date' if 'Date' in df.columns else 'Datetime'
        
        # Clean dataframe for our ML model
        df_clean = pd.DataFrame({
            "Date": pd.to_datetime(df[date_col]).dt.tz_localize(None),
            "Price": df["Close"],
            "Volume": df["Volume"]
        })
        return df_clean
    except Exception as e:
        return pd.DataFrame(columns=["Date", "Price", "Volume"])

st.info("Fetching live historical data from Yahoo Finance...")
coin_data_dict = {}

for coin_name in selected_coins:
    symbol = coins_dict.get(coin_name)
    if symbol:
        coin_data_dict[coin_name] = fetch_yf_history(symbol, timeframe)

st.success("Data fetching complete!")

# ---- FEATURE ENGINEERING + ML ----
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
        
