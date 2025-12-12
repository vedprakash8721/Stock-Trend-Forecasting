# app.py
import os
import io
import math
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="Stock Predictor (LSTM)", layout="wide")
pio.templates.default = "plotly_dark"  # set plotly dark theme globally

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.title("⚙️ Configuration")

default_ticker = "POWERGRID.NS"
stock = st.sidebar.text_input("Stock Ticker (e.g., AAPL, RELIANCE.NS)", value=default_ticker).strip().upper()

col_dates = st.sidebar.columns(2)
with col_dates[0]:
    start = st.date_input("Start Date", value=dt.date(2015, 1, 1))
with col_dates[1]:
    end = st.date_input("End Date", value=dt.date.today())

seq_len = st.sidebar.number_input("Sequence Length (days)", min_value=20, max_value=300, value=100, step=5)
train_split = st.sidebar.slider("Train Split (%)", min_value=50, max_value=90, value=70, step=5)

BASE_DIR = os.path.dirname(__file__)
default_model_path = os.path.join(BASE_DIR, "stock_dl_model.h5")
model_path = st.sidebar.text_input("Model Path (.h5)", value=default_model_path)
uploaded_model = st.sidebar.file_uploader("...or Upload Keras Model (.h5)", type=["h5"])

st.sidebar.markdown("---")
st.sidebar.markdown("Advanced options (optional)")
use_risk_free = st.sidebar.checkbox("Show simple backtest (strategy) & metrics", value=True)
run_btn = st.sidebar.button("🚀 Run Inference")

st.title("📈 Stock Price Prediction — LSTM")
st.caption("Interactive dashboard — Plotly candlestick, EMAs, prediction vs actual, metrics, downloads and simple explainability.")

# ---------------------------
# Utilities
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_keras_model(path_or_bytes):
    # Accept filepath or uploaded file-like
    try:
        if hasattr(path_or_bytes, "read"):
            data = path_or_bytes.read()
            tmp = os.path.join(BASE_DIR, "uploaded_model.h5")
            with open(tmp, "wb") as f:
                f.write(data)
            return load_model(tmp)
        else:
            return load_model(path_or_bytes)
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")

@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def fig_to_png_bytes_plotly(fig):
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        return io.BytesIO(img_bytes)
    except Exception:
        # fallback: render to PNG via static image (may require kaleido)
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        return buf

# ---------------------------
# Load model (if provided)
# ---------------------------
model = None
if uploaded_model:
    try:
        model = load_keras_model(uploaded_model)
    except Exception as e:
        st.sidebar.error(f"Uploaded model load failed: {e}")
elif os.path.exists(model_path):
    try:
        model = load_keras_model(model_path)
    except Exception as e:
        st.sidebar.error(f"Model load failed from path: {e}")

# If model not present, warn but allow data exploration
if model is None:
    st.sidebar.warning("Model not loaded — upload or provide valid .h5 path to run predictions.")
    # Do not stop; allow user to explore EMAs etc.
    
# ---------------------------
# Run button action
# ---------------------------
if run_btn:
    with st.spinner("Downloading market data..."):
        df = fetch_data(stock, start, end)

    if df is None or df.empty:
        st.error("No data returned. Check ticker and date range.")
        st.stop()

    # Basic cleaning
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()

    st.subheader(f"Overview — {stock}")
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        st.dataframe(df.tail(8))
    with c2:
        st.markdown("**Descriptive Stats (Close)**")
        st.dataframe(df["Close"].describe().to_frame())

    # EMAs
    close = df["Close"].copy()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # Plotly Candlestick with EMA overlays
    st.subheader("Interactive Candlestick — Close + EMAs")
    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"))
    fig_candle.add_trace(go.Scatter(x=ema20.index, y=ema20.values, mode="lines", name="EMA20", line=dict(width=1.5)))
    fig_candle.add_trace(go.Scatter(x=ema50.index, y=ema50.values, mode="lines", name="EMA50", line=dict(width=1.5)))
    fig_candle.add_trace(go.Scatter(x=ema100.index, y=ema100.values, mode="lines", name="EMA100", line=dict(width=1)))
    fig_candle.add_trace(go.Scatter(x=ema200.index, y=ema200.values, mode="lines", name="EMA200", line=dict(width=1)))
    fig_candle.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=1.02))
    fig_candle.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_candle, use_container_width=True)

    # ---------------------------
    # If model not loaded, stop after EMAs (user can still download data)
    # ---------------------------
    if model is None:
        st.info("Model not loaded — upload a .h5 model in the sidebar to run predictions.")
        # allow CSV download of raw data
        csv_buf = io.StringIO()
        df.to_csv(csv_buf)
        st.download_button("⬇️ Download Raw Data (CSV)", csv_buf.getvalue(), file_name=f"{stock}_raw.csv", mime="text/csv")
        st.stop()

    # ---------------------------
    # Prepare data for inference
    # ---------------------------
    split_idx = int(len(df) * (train_split / 100.0))
    close_series = df["Close"]
    data_training = pd.DataFrame(close_series.iloc[:split_idx])
    data_testing = pd.DataFrame(close_series.iloc[split_idx:])

    if len(data_testing) < seq_len:
        st.error(f"Test size {len(data_testing)} smaller than seq_len {seq_len}. Adjust parameters.")
        st.stop()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_training)  # fit only on training data

    past_window = data_training.tail(seq_len)
    final_df = pd.concat([past_window, data_testing], ignore_index=True)
    input_scaled = scaler.transform(final_df)

    x_test, y_test = [], []
    for i in range(seq_len, input_scaled.shape[0]):
        x_test.append(input_scaled[i - seq_len:i])
        y_test.append(input_scaled[i, 0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    if x_test.ndim == 2:
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Predict
    with st.spinner("Running model inference..."):
        y_pred_scaled = model.predict(x_test, verbose=0)
    try:
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    except Exception:
        # fallback: if scaler expects 2D arrays shaped differently
        y_pred = y_pred_scaled.reshape(-1, 1)
        y_true = y_test.reshape(-1, 1)

    # Align dates for plotting
    pred_dates = data_testing.index[-len(y_true):]

    # ---------------------------
    # Prediction vs Actual (interactive)
    # ---------------------------
    st.subheader("Prediction vs Actual")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=y_true.flatten(), mode="lines", name="Actual", line=dict(color="cyan", width=2)))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=y_pred.flatten(), mode="lines", name="Predicted", line=dict(color="magenta", width=2)))
    # residuals
    resid = y_true.flatten() - y_pred.flatten()
    fig_pred.add_trace(go.Bar(x=pred_dates, y=resid, name="Residuals", opacity=0.25, marker=dict(color="grey")))
    fig_pred.update_layout(height=420, legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig_pred, use_container_width=True)

    # ---------------------------
    # Metrics
    # ---------------------------
    mae = float(np.mean(np.abs(y_true.flatten() - y_pred.flatten())))
    rmse = float(np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2)))
    with st.container():
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("MAE", f"{mae:,.4f}")
        c2.metric("RMSE", f"{rmse:,.4f}")
        c3.markdown("**Notes:** MAE/RMSE computed on test window (inverse-transformed prices).")

    # ---------------------------
    # Explainability: rolling error + residual distribution
    # ---------------------------
    st.subheader("Explainability & Residuals")
    roll_w = min(20, len(resid))
    rolling_rmse = pd.Series(np.abs(resid)).rolling(window=roll_w).mean()
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=pred_dates, y=rolling_rmse, mode="lines", name=f"Rolling MAE ({roll_w})"))
    fig_res.add_trace(go.Histogram(x=resid, nbinsx=40, name="Residual dist", opacity=0.6))
    fig_res.update_layout(barmode="overlay", height=380)
    st.plotly_chart(fig_res, use_container_width=True)

    # ---------------------------
    # Simple Backtest (signal: predict > prev_close => long next day)
    # ---------------------------
    if use_risk_free:
        st.subheader("Simple Backtest — Signal: Predicted > Previous Close => Long 1 day")

        # Build previous close series aligned with predictions
        prev_close = data_testing["Close"].values[-len(y_pred):]
        # Signal: if predicted price > prev_close then long, else flat
        signals = (y_pred.flatten() > prev_close).astype(int)

        # compute daily returns for the period (next day close / today close - 1)
        test_index = data_testing.index[-len(y_pred):]
        # to compute returns we need next-day close; use df index to get actual next close where available
        returns = []
        for dt_i in range(len(test_index)):
            try:
                today = test_index[dt_i]
                # next day index safe guard
                next_idx = df.index.get_loc(today) + 1
                if next_idx < len(df):
                    nxt_close = df["Close"].iloc[next_idx]
                    today_close = df["Close"].iloc[df.index.get_loc(today)]
                    ret = (nxt_close / today_close) - 1.0
                else:
                    ret = 0.0
            except Exception:
                ret = 0.0
            returns.append(ret)
        returns = np.array(returns)

        strat_returns = signals * returns
        cum_strat = (1 + strat_returns).cumprod() - 1
        cum_buyhold = (1 + returns).cumprod() - 1

        bt_df = pd.DataFrame({
            "date": test_index,
            "signal": signals,
            "daily_return": returns,
            "strategy_return": strat_returns,
            "cum_strategy": cum_strat,
            "cum_buyhold": cum_buyhold
        })

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["cum_strategy"], mode="lines", name="Strategy Cumulative"))
        fig_bt.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["cum_buyhold"], mode="lines", name="Buy & Hold"))
        fig_bt.update_layout(height=380)
        st.plotly_chart(fig_bt, use_container_width=True)

        # Backtest summary
        final_strat = float(bt_df["cum_strategy"].iloc[-1])
        final_bh = float(bt_df["cum_buyhold"].iloc[-1])
        st.write(f"Strategy final return: **{final_strat*100:.2f}%**  — Buy & Hold: **{final_bh*100:.2f}%**")

    # ---------------------------
    # Downloads: raw CSV, preds CSV, PNGs
    # ---------------------------
    st.subheader("Downloads")
    # raw data
    csv_buf = io.StringIO()
    df.to_csv(csv_buf)
    st.download_button("⬇️ Download Raw Data (CSV)", csv_buf.getvalue(), file_name=f"{stock}_raw.csv", mime="text/csv")

    # predictions CSV
    pred_df = pd.DataFrame({"date": pred_dates, "actual": y_true.flatten(), "predicted": y_pred.flatten()})
    st.download_button("⬇️ Download Predictions (CSV)", pred_df.to_csv(index=False), file_name=f"{stock}_preds.csv", mime="text/csv")

    # download plotly png (prediction)
    try:
        png_buf = fig_pred.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button("⬇️ Download Prediction Plot (PNG)", png_buf, file_name=f"{stock}_pred_plot.png", mime="image/png")
    except Exception:
        # fallback using fig_to_png_bytes_plotly
        png_buf = fig_to_png_bytes_plotly(fig_pred)
        st.download_button("⬇️ Download Prediction Plot (PNG)", png_buf.getvalue(), file_name=f"{stock}_pred_plot.png", mime="image/png")

    st.success("Inference complete.")

else:
    st.info("Set parameters on the left and click **Run Inference** to generate predictions.")
    st.markdown(
        """
        ** Developed by ~ Abhishek Kushwaha **
        """
    )
