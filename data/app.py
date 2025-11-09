# app.py
import os
import io
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Reduce TF log noise (optional, set before importing TF/Keras in larger apps)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="Stock Predictor (LSTM)", layout="wide")
plt.style.use("fivethirtyeight")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.title("⚙️ Configuration")

default_ticker = "POWERGRID.NS"
stock = st.sidebar.text_input("Stock Ticker (e.g., AAPL, RELIANCE.NS)", value=default_ticker).strip()

col_dates = st.sidebar.columns(2)
with col_dates[0]:
    start = st.date_input("Start Date", value=dt.date(2000, 1, 1))
with col_dates[1]:
    end = st.date_input("End Date", value=dt.date(2024, 10, 1))

seq_len = st.sidebar.number_input("Sequence Length (days)", min_value=20, max_value=300, value=100, step=5)
train_split = st.sidebar.slider("Train Split (%)", min_value=50, max_value=90, value=70, step=5)

# Model path handling: default to same folder as this script
BASE_DIR = os.path.dirname(__file__)
default_model_path = os.path.join(BASE_DIR, "stock_dl_model.h5")

model_path = st.sidebar.text_input("Model Path (.h5)", value=default_model_path)
uploaded_model = st.sidebar.file_uploader("...or Upload Keras Model (.h5)", type=["h5"])

run_btn = st.sidebar.button("🚀 Run Inference")

st.title("📈 Stock Price Prediction — LSTM (Streamlit)")
st.caption("Interactive app using your trained Keras model for next-day close predictions from rolling windows.")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_keras_model(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray, io.BytesIO)):
        tmp = os.path.join(BASE_DIR, "uploaded_model.h5")
        with open(tmp, "wb") as f:
            f.write(path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes.getbuffer())
        return load_model(tmp)
    else:
        return load_model(path_or_bytes)

model = None
if uploaded_model is not None:
    model = load_keras_model(uploaded_model)
elif os.path.exists(model_path):
    model = load_keras_model(model_path)

if model is None:
    st.warning("Model not found. Upload a `.h5` model or set a valid path.")
    st.stop()

# ---------------------------
# Fetch Data
# ---------------------------
@st.cache_data(show_spinner=True)
def fetch_data(ticker: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    return df

if run_btn:
    with st.spinner("Downloading market data..."):
        df = fetch_data(stock, start, end)

    if df.empty:
        st.error("No data returned. Check ticker or date range.")
        st.stop()

    st.subheader(f"Overview: {stock}")
    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.dataframe(df.tail(10))
    with c2:
        st.markdown("**Descriptive Stats**")
        st.dataframe(df.describe())

    # ---------------------------
    # EMAs
    # ---------------------------
    close = df["Close"].copy()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # ---------------------------
    # Train/Test Split
    # ---------------------------
    split_idx = int(len(df) * (train_split / 100.0))
    data_training = pd.DataFrame(close.iloc[:split_idx])
    data_testing = pd.DataFrame(close.iloc[split_idx:])

    # Guardrail: ensure enough test samples for chosen window
    if len(data_testing) < seq_len:
        st.error(
            f"Not enough test points ({len(data_testing)}) for seq_len={seq_len}. "
            f"Reduce Sequence Length or extend End Date / lower Train Split."
        )
        st.stop()

    # ---------------------------
    # Scaling (FIT on TRAIN only; TRANSFORM others)  ✅ no leakage
    # ---------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    _ = scaler.fit_transform(data_training)  # fit on train

    # Context window from end of training
    past_window = data_training.tail(seq_len)
    final_df = pd.concat([past_window, data_testing], axis=0, ignore_index=True)

    # Transform only (no refit)
    input_scaled = scaler.transform(final_df)

    # ---------------------------
    # Build x_test / y_test
    # ---------------------------
    x_test, y_test = [], []
    for i in range(seq_len, input_scaled.shape[0]):
        x_test.append(input_scaled[i - seq_len:i])
        y_test.append(input_scaled[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # LSTM expects 3D [samples, timesteps, features]
    if x_test.ndim == 2:
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # ---------------------------
    # Predict (scaled) -> inverse transform to original scale
    # ---------------------------
    with st.spinner("Running model inference..."):
        y_pred_scaled = model.predict(x_test, verbose=0)

    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ---------------------------
    # Charts
    # ---------------------------
    st.subheader("Price with EMAs")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(close.index, close.values, label="Close", linewidth=1.2)
    ax1.plot(ema20.index, ema20.values, label="EMA 20")
    ax1.plot(ema50.index, ema50.values, label="EMA 50")
    ax1.set_title(f"{stock} — Close with 20/50 EMA")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Price")
    ax1.legend()
    st.pyplot(fig1, clear_figure=True)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(close.index, close.values, label="Close", linewidth=1.2)
    ax2.plot(ema100.index, ema100.values, label="EMA 100")
    ax2.plot(ema200.index, ema200.values, label="EMA 200")
    ax2.set_title(f"{stock} — Close with 100/200 EMA")
    ax2.set_xlabel("Date"); ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

    # ---------------------------
    # Prediction vs Original — robust alignment
    # ---------------------------
    st.subheader("Prediction vs Original (Test Segment)")

    pred_dates = data_testing.index  # base index for test segment

    # Safety alignment to avoid length mismatches
    n = min(len(pred_dates), len(y_true), len(y_pred))
    if n == 0:
        st.error("No test samples to plot. Adjust date range or Sequence Length.")
        st.stop()

    pred_dates = pred_dates[-n:]
    y_true_plot = y_true.flatten()[-n:]
    y_pred_plot = y_pred.flatten()[-n:]

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(pred_dates, y_true_plot, label="Original Price", linewidth=1.3)
    ax3.plot(pred_dates, y_pred_plot, label="Predicted Price", linewidth=1.3)
    ax3.set_title("Prediction vs Original")
    ax3.set_xlabel("Date"); ax3.set_ylabel("Price")
    ax3.legend()
    st.pyplot(fig3, clear_figure=True)

    # ---------------------------
    # Metrics
    # ---------------------------
    mae = float(np.mean(np.abs(y_true_plot - y_pred_plot)))
    rmse = float(np.sqrt(np.mean((y_true_plot - y_pred_plot) ** 2)))
    st.markdown(f"**MAE:** {mae:,.4f} &nbsp;&nbsp; **RMSE:** {rmse:,.4f}")

    # ---------------------------
    # Downloads
    # ---------------------------
    csv_buf = io.StringIO()
    df.to_csv(csv_buf)
    st.download_button(
        label=f"⬇️ Download {stock} dataset (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"{stock}_dataset.csv",
        mime="text/csv",
    )

    # Predictions CSV
    pred_df = pd.DataFrame({
        "date": pred_dates,
        "actual": y_true_plot,
        "predicted": y_pred_plot
    })
    st.download_button(
        "⬇️ Download Predictions (CSV)",
        data=pred_df.to_csv(index=False),
        file_name=f"{stock}_predictions.csv",
        mime="text/csv",
    )

    # Plot PNG
    def fig_to_png_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        return buf

    st.download_button(
        "⬇️ Download Prediction Plot (PNG)",
        data=fig_to_png_bytes(fig3),
        file_name=f"{stock}_prediction.png",
        mime="image/png",
    )

    st.success("Inference complete.")

else:
    st.info("Set parameters on the left and click **Run Inference** to generate predictions.")
    st.markdown(
        """
        **Notes**
        - Place your trained model at `stock_dl_model.h5` (same folder as this file) or upload via the sidebar.  
        - For NSE tickers, use the `.NS` suffix (e.g., `RELIANCE.NS`, `POWERGRID.NS`).  
        - The scaler is fit on the train split only (no leakage). Test windows are built with a past context window.
        """
    )
