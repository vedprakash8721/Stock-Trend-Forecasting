# app.py
import os
import io
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# ---------------------------
# Global config
# ---------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="Stock Trend Forecasting ", layout="wide")

# IMPORTANT: Fix hover & legend visibility
pio.templates.default = "plotly_white"

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("⚙️ Configuration")

# ---------------------------
# Mode Selection
# ---------------------------
mode = st.sidebar.radio(
    "Select Mode",
    ["📊 Market Guide", "Learning Mode"]
)

stock = st.sidebar.text_input(
    "Use standard US stock tickers (e.g., AAPL for Apple, MSFT for Microsoft, GOOG for Google, AMZN for Amazon, TSLA for Tesla, NVDA for NVIDIA,META for Meta,NFLX for Netflix,KO for Coca-Cola, etc.)",
    value="POWERGRID.NS"
).strip().upper()

col_dates = st.sidebar.columns(2)
with col_dates[0]:
    start = st.date_input("Start Date", value=dt.date(2015, 1, 1))
with col_dates[1]:
    end = st.date_input("End Date", value=dt.date.today())

seq_len = st.sidebar.number_input("Sequence Length (days)", 20, 300, 100, 5)
train_split = st.sidebar.slider("Train Split (%)", 50, 90, 70, 5)
recent_days = st.sidebar.slider("Recent days to show", 5, 100, 30)

BASE_DIR = os.path.dirname(__file__)
model_path = st.sidebar.text_input(
    "Model Path (.h5)",
    value=os.path.join(BASE_DIR, "stock_dl_model.h5")
)
uploaded_model = st.sidebar.file_uploader("Upload Keras Model (.h5)", type=["h5"])

run_btn = st.sidebar.button("🚀 Run Inference")

# ---------------------------
# Title
# ---------------------------
st.title("📈 Stock Trend Forecasting")
st.caption("Market context + ML predictions + downloadable datasets")

# ---------------------------
# Utilities
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_keras_model(path_or_file):
    if hasattr(path_or_file, "read"):
        tmp_path = os.path.join(BASE_DIR, "uploaded_model.h5")
        with open(tmp_path, "wb") as f:
            f.write(path_or_file.read())
        return load_model(tmp_path)
    return load_model(path_or_file)

@st.cache_data(show_spinner=False)
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ---------------------------
# Load model
# ---------------------------
model = None
try:
    if uploaded_model:
        model = load_keras_model(uploaded_model)
    elif os.path.exists(model_path):
        model = load_keras_model(model_path)
except Exception as e:
    st.sidebar.error(f"Model load failed: {e}")

# ---------------------------
# Run app
# ---------------------------
if mode == "📊 Market Guide" and run_btn:
    df = fetch_data(stock, start, end)

    if df.empty:
        st.error("No data returned. Check ticker or date range.")
        st.stop()

    df = df.dropna().sort_index()

    # ===========================
    # Market data tables
    # ===========================
    st.subheader("📋 Recent Market Data (OHLCV)")
    recent_df = df.tail(recent_days)
    st.dataframe(recent_df, use_container_width=True)

    st.download_button(
        "⬇️ Download Recent OHLCV (CSV)",
        recent_df.to_csv(),
        file_name=f"{stock}_recent_ohlcv.csv",
        mime="text/csv"
    )

    st.subheader("📊 Descriptive Statistics (Close Price)")
    st.dataframe(df["Close"].describe().to_frame())

    # ===========================
    # Candlestick + EMA
    # ===========================
    st.markdown("---")
    st.subheader("📉 Candlestick with EMAs")

    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC"
    ))

    for span in [20, 50, 100, 200]:
        fig_candle.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"].ewm(span=span).mean(),
            mode="lines",
            name=f"EMA {span}"
        ))

    fig_candle.update_layout(
        height=520,
        legend=dict(orientation="h", y=1.02)
    )

    st.plotly_chart(fig_candle, use_container_width=True)

    # ===========================
    # Model check
    # ===========================
    if model is None:
        st.warning("Model not loaded. Upload a .h5 model to run predictions.")

        st.download_button(
            "⬇️ Download Full Raw Data (CSV)",
            df.to_csv(),
            file_name=f"{stock}_raw_data.csv",
            mime="text/csv"
        )
        st.stop()

    # ===========================
    # Prepare data
    # ===========================
    close = df["Close"]
    split_idx = int(len(close) * train_split / 100)

    train = close[:split_idx]
    test = close[split_idx:]

    if len(test) < seq_len:
        st.error("Test data smaller than sequence length.")
        st.stop()

    scaler = MinMaxScaler()
    scaler.fit(train.values.reshape(-1, 1))

    combined = pd.concat([train.tail(seq_len), test])
    scaled = scaler.transform(combined.values.reshape(-1, 1))

    x_test, y_test = [], []
    for i in range(seq_len, len(scaled)):
        x_test.append(scaled[i-seq_len:i])
        y_test.append(scaled[i])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # ===========================
    # Prediction
    # ===========================
    y_pred = model.predict(x_test, verbose=0)

    y_pred = scaler.inverse_transform(y_pred)
    y_true = scaler.inverse_transform(y_test)

    dates = test.index[-len(y_true):]

    # ===========================
    # Prediction plot
    # ===========================
    st.markdown("---")
    st.subheader(" Prediction vs Actual")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=dates,
        y=y_true.flatten(),
        mode="lines",
        name="Actual"
    ))
    fig_pred.add_trace(go.Scatter(
        x=dates,
        y=y_pred.flatten(),
        mode="lines",
        name="Predicted"
    ))

    fig_pred.update_layout(
        height=420,
        legend=dict(orientation="h", y=1.02)
    )

    st.plotly_chart(fig_pred, use_container_width=True)

    # ===========================
    # Metrics
    # ===========================
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    # ---------------------------
    # Market Summary Cards
    # ---------------------------
    # Market Summary (Professional UI)
    # ---------------------------
    st.markdown("---")
    st.subheader("📊 Market Summary")

    current_price = df["Close"].iloc[-1]
    predicted_price = y_pred[-1][0]
    change_percent = ((predicted_price - current_price) / current_price) * 100

    # Create container for clean grouping
    with st.container():

        col1, col2, col3 = st.columns(3)

        # Current Price
        with col1:
            st.metric(
                label="📍 Current Price",
                value=f"₹{current_price:.2f}"
            )

        # Predicted Price with color change
        with col2:
            st.metric(
                label="🔮 Predicted Price",
                value=f"₹{predicted_price:.2f}",
                delta=f"{change_percent:.2f}%"
            )

        # Market Direction
        with col3:
            if change_percent > 0:
                st.success("📈 Bullish Trend")
            elif change_percent < 0:
                st.error("📉 Bearish Trend")
            else:
                st.warning("📊 Sideways Market")

    # ---------------------------
    # ---------------------------
    # Trading Signal (Professional UI)
    # ---------------------------
    st.markdown("---")
    st.subheader("📢 Trading Signal")

    # Signal logic (keep yours if already defined)
    if change_percent > 1:
        signal = "BUY"
    elif change_percent < -1:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Center alignment using columns
    left, center, right = st.columns([1, 2, 1])

    with center:
        if signal == "BUY":
            st.success("🟢 BUY")
        elif signal == "SELL":
            st.error("🔴 SELL")
        else:
            st.warning("🟡 HOLD")

    # ---------------------------
    # Risk Level (Professional UI)
    # ---------------------------
    st.subheader("⚠️ Risk Level")

    # Risk estimation
    volatility = df["Close"].pct_change().std()

    if volatility < 0.01:
        risk = "LOW"
    elif volatility < 0.02:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    # Center alignment (same style as signal)
    left, center, right = st.columns([1, 2, 1])

    with center:
        if risk == "LOW":
            st.success("🟢 LOW RISK")
        elif risk == "MEDIUM":
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")
    # ---------------------------
    # Explanation (WHY section)
    # ---------------------------
    st.markdown("---")
    st.subheader(" Market Trend ")

    # Signal explanation
    if signal == "BUY":
        st.success(
            "📈 The model predicts price increase with positive momentum. "
            "This indicates a potential buying opportunity."
        )
    elif signal == "SELL":
        st.error(
            "📉 The model predicts price decline. "
            "Market may face downward pressure, suggesting caution or selling."
        )
    else:
        st.warning(
            "📊 The market is moving sideways with no strong trend. "
            "Holding position is safer."
        )

    # Risk explanation
    if risk == "HIGH":
        st.info(
            "⚠️ High volatility detected. Price may fluctuate rapidly, increasing risk."
        )
    elif risk == "MEDIUM":
        st.info(
            "⚠️ Moderate volatility. Some uncertainty exists in price movement."
        )
    else:
        st.info(
            "✅ Low volatility. Market conditions are relatively stable."
        )
    # ---------------------------
    # ---------------------------
    # Smart AI Insights (Upgraded)
    # ---------------------------
    st.markdown("---")
    st.subheader(" AI Insights")

    col1, col2 = st.columns(2)

    # ---------------- LEFT: Market Behavior ----------------
    with col1:
        st.markdown("### 📊 Market Behavior")

        if change_percent > 1:
            st.success("📈 Strong bullish momentum detected")
        elif change_percent < -1:
            st.error("📉 Strong bearish pressure detected")
        else:
            st.warning("📊 Sideways movement (low momentum)")

        st.caption(f"Predicted change: {change_percent:.2f}%")

    # ---------------- RIGHT: Volatility Insight ----------------
    with col2:
        st.markdown("### ⚠️ Volatility Analysis")

        if volatility > 0.02:
            st.error("High volatility → Risky environment")
        elif volatility > 0.01:
            st.warning("Moderate volatility → Uncertain market")
        else:
            st.success("Low volatility → Stable conditions")

    # ---------------- EXPANDABLE PRO INSIGHT ----------------
    with st.expander("📌 Detailed AI Explanation"):
        
        if signal == "BUY":
            st.write(
                "The model identifies upward trend supported by price movement and momentum. "
                "Combined with current conditions, this suggests a potential buying opportunity."
            )
        elif signal == "SELL":
            st.write(
                "The model detects weakening price trend. "
                "Market may experience downward correction or selling pressure."
            )
        else:
            st.write(
                "The model does not detect a strong directional trend. "
                "Market is likely consolidating."
            )

        st.write(
            "Volatility analysis is used to estimate risk. "
            "Higher volatility indicates unpredictable price swings."
        )
        
    # ===========================
    # Downloads
    # ===========================
    st.markdown("---")
    st.subheader(" Downloads")

    st.download_button(
        "⬇️ Download Full Raw Data (CSV)",
        df.to_csv(),
        file_name=f"{stock}_raw_data.csv",
        mime="text/csv"
    )

    pred_df = pd.DataFrame({
        "date": dates,
        "actual": y_true.flatten(),
        "predicted": y_pred.flatten()
    })

    st.download_button(
        "⬇️ Download Predictions (CSV)",
        pred_df.to_csv(index=False),
        file_name=f"{stock}_predictions.csv",
        mime="text/csv"
    )

    st.download_button(
        "⬇️ Download Prediction Plot (HTML)",
        fig_pred.to_html(full_html=False, include_plotlyjs="cdn"),
        file_name=f"{stock}_prediction_plot.html",
        mime="text/html"
    )

    st.success("✅ Analysis complete. Review market summary and insights above.")

elif mode == "📊 Market Guide":
    st.info("Set parameters and click **Run Inference**.")

# ---------------------------
# Learning Mode
# ---------------------------
if mode == "📘 Learning Mode":
    st.title("📘 Stock Market Learning Hub")
    st.info("Select a topic from below to start learning.")
# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    ---
    **Developed by ~ Abhishek Kushwaha**
    """
)
