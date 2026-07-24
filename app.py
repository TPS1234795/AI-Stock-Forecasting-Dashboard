# ==========================================================
# AI-POWERED STOCK FORECASTING DASHBOARD
# Hybrid CNN + LSTM
# ==========================================================

# ==========================================================
# STEP 1 : IMPORT REQUIRED LIBRARIES
# ==========================================================
import os
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model


# ==========================================================
# STEP 2 : CONFIGURE STREAMLIT PAGE
# ==========================================================
st.set_page_config(
    page_title="AI Stock Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOOKBACK = 60          # days the model looks back on
FORECAST_HORIZON = 5   # days to iteratively forecast ahead


# ==========================================================
# STEP 3 : LOAD CUSTOM CSS
# ==========================================================
def load_css():
    css_path = os.path.join(BASE_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Custom CSS not found at {css_path} — using default theme.")

load_css()


# ==========================================================
# STEP 4 : LOAD MODEL & SCALER (cached so it only loads once)
# ==========================================================
@st.cache_resource(show_spinner="Loading CNN-LSTM model...")
def load_artifacts():
    model_path = os.path.join(BASE_DIR, "models", "cnn_lstm_model.keras")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_artifacts()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    model, scaler = None, None
    st.error(f"❌ Could not load model/scaler: {e}")


# ==========================================================
# STEP 5 : DATA HELPERS (cached so switching stocks is fast)
# ==========================================================
@st.cache_data(ttl=3600, show_spinner="Fetching latest market data...")
def get_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def forecast_future(close_series: pd.Series, model, scaler, horizon: int = FORECAST_HORIZON):
    """Iteratively rolls the model's own predictions forward to forecast several days ahead."""
    scaled = scaler.transform(close_series.values.reshape(-1, 1)).flatten()
    window = list(scaled[-LOOKBACK:])
    preds_scaled = []

    for _ in range(horizon):
        x = np.reshape(np.array(window[-LOOKBACK:]), (1, LOOKBACK, 1))
        next_scaled = model.predict(x, verbose=0)[0][0]
        preds_scaled.append(next_scaled)
        window.append(next_scaled)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds


# ==========================================================
# STEP 6 : HEADER
# ==========================================================
st.markdown("""
<div class="hero">
    <h1>📈 AI-Powered Stock Forecasting Dashboard</h1>
    <p>Hybrid <b>CNN + LSTM</b> deep learning model for next-day &amp; short-term price prediction</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# STEP 7 : SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">📊 StockForecastAI</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Stock Selection")
    stocks = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]
    selected_stock = st.selectbox("Choose a Stock", stocks)

    period = st.select_slider(
        "History Range",
        options=["6mo", "1y", "2y", "5y", "max"],
        value="5y"
    )

    show_sma = st.checkbox("Show Moving Averages (SMA 20/50)", value=True)
    show_volume = st.checkbox("Show Volume", value=True)

    st.markdown("---")
    st.subheader("Project Info")
    st.info(
        "**Model:** CNN + LSTM\n\n"
        "**Framework:** TensorFlow / Keras\n\n"
        "**Data Source:** Yahoo Finance\n\n"
        f"**Lookback Window:** {LOOKBACK} days\n\n"
        "**Predicts:** Next closing price(s)"
    )

    st.markdown("---")
    st.caption(f"🕒 Last refreshed: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ==========================================================
# STEP 8 : LOAD DATA (with error handling)
# ==========================================================
try:
    stock_data = get_stock_data(selected_stock, period)
    if stock_data.empty:
        st.error(f"No data returned for {selected_stock}. Try again later.")
        st.stop()
    stock_data = add_technical_indicators(stock_data)
except Exception as e:
    st.error(f"❌ Failed to fetch data for {selected_stock}: {e}")
    st.stop()


# ==========================================================
# STEP 9 : TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Prediction", "📄 Historical Data"])

# ----------------------------------------------------------
# TAB 1 : DASHBOARD
# ----------------------------------------------------------
with tab1:
    latest = stock_data.iloc[-1]
    prev = stock_data.iloc[-2] if len(stock_data) > 1 else latest
    change = float(latest["Close"]) - float(prev["Close"])
    pct_change = (change / float(prev["Close"])) * 100 if float(prev["Close"]) != 0 else 0

    st.markdown(f"### {selected_stock} Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Open", f"${float(latest['Open']):.2f}")
    col2.metric("High", f"${float(latest['High']):.2f}")
    col3.metric("Low", f"${float(latest['Low']):.2f}")
    col4.metric("Close", f"${float(latest['Close']):.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
    col5.metric("Volume", f"{float(latest['Volume']):,.0f}")

    st.markdown("---")

    # Candlestick + volume chart
    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1.0]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.03
    )

    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data["Open"], high=stock_data["High"],
        low=stock_data["Low"], close=stock_data["Close"],
        name="Price",
        increasing_line_color="#00CC96",
        decreasing_line_color="#EF553B"
    ), row=1, col=1)

    if show_sma:
        fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data["SMA20"],
            name="SMA 20", line=dict(color="#FFA15A", width=1.5)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data["SMA50"],
            name="SMA 50", line=dict(color="#636EFA", width=1.5)
        ), row=1, col=1)

    if show_volume:
        colors = np.where(stock_data["Close"] >= stock_data["Open"], "#00CC96", "#EF553B")
        fig.add_trace(go.Bar(
            x=stock_data.index, y=stock_data["Volume"],
            name="Volume", marker_color=colors, opacity=0.6
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=False,
        title=f"{selected_stock} Price Chart ({period})",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=60, b=20, l=10, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI chart
    with st.expander("📉 RSI (Relative Strength Index)"):
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data["RSI"],
            line=dict(color="#00CC96", width=2), name="RSI"
        ))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="#EF553B", annotation_text="Overbought")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="#00CC96", annotation_text="Oversold")
        rsi_fig.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20))
        st.plotly_chart(rsi_fig, use_container_width=True)


# ----------------------------------------------------------
# TAB 2 : AI PREDICTION
# ----------------------------------------------------------
with tab2:
    st.subheader("🤖 AI Stock Price Prediction")

    if not MODEL_READY:
        st.error("Model is not available. Prediction cannot be performed.")
    else:
        try:
            close_data = stock_data[["Close"]].dropna()

            if len(close_data) < LOOKBACK:
                st.warning(f"Need at least {LOOKBACK} days of data to predict. Only {len(close_data)} available.")
            else:
                with st.spinner("Running CNN-LSTM inference..."):
                    scaled_data = scaler.transform(close_data)
                    last_window = scaled_data[-LOOKBACK:]
                    X_test = np.reshape(last_window, (1, LOOKBACK, 1))
                    next_pred_scaled = model.predict(X_test, verbose=0)
                    predicted_price = float(scaler.inverse_transform(next_pred_scaled)[0][0])
                    future_prices = forecast_future(close_data["Close"], model, scaler, FORECAST_HORIZON)

                latest_close = float(close_data.iloc[-1])
                delta = predicted_price - latest_close
                pct = (delta / latest_close) * 100 if latest_close else 0

                col1, col2, col3 = st.columns(3)
                col1.metric("Latest Closing Price", f"${latest_close:.2f}")
                col2.metric("Predicted Next Close", f"${predicted_price:.2f}", f"{delta:+.2f} ({pct:+.2f}%)")
                trend = "📈 Bullish" if delta > 0 else "📉 Bearish" if delta < 0 else "➖ Flat"
                col3.metric("Signal", trend)

                st.markdown("---")

                # Historical (last 60d) + forecast overlay chart
                future_dates = pd.bdate_range(
                    start=stock_data.index[-1] + pd.Timedelta(days=1),
                    periods=FORECAST_HORIZON
                )

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=stock_data.index[-LOOKBACK:], y=close_data["Close"].tail(LOOKBACK),
                    mode="lines", name="Last 60 Days (Actual)",
                    line=dict(color="#00CC96", width=3)
                ))
                fig2.add_trace(go.Scatter(
                    x=[stock_data.index[-1]] + list(future_dates),
                    y=[latest_close] + list(future_prices),
                    mode="lines+markers", name=f"{FORECAST_HORIZON}-Day Forecast",
                    line=dict(color="#FFA15A", width=3, dash="dash"),
                    marker=dict(size=7)
                ))
                fig2.update_layout(
                    template="plotly_dark", height=480,
                    xaxis_title="Date", yaxis_title="Price (USD)",
                    title=f"{selected_stock}: Actual vs Forecasted Closing Price",
                    legend=dict(orientation="h", y=1.05, x=0)
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Forecast table
                st.markdown("##### 5-Day Forecast Detail")
                forecast_df = pd.DataFrame({
                    "Date": future_dates.strftime("%d %b %Y"),
                    "Predicted Close ($)": np.round(future_prices, 2)
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.success(
                    f"📈 Predicted closing price for the next trading day: **${predicted_price:.2f}** "
                    f"({pct:+.2f}% vs. last close)."
                )
                st.caption(
                    "⚠️ This is a statistical model output based on historical prices only. "
                    "It is not financial advice."
                )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


# ----------------------------------------------------------
# TAB 3 : HISTORICAL DATA
# ----------------------------------------------------------
with tab3:
    st.subheader("📄 Historical Stock Data")
    st.dataframe(stock_data, use_container_width=True)

    st.markdown("---")
    csv = stock_data.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Historical Data (CSV)",
        data=csv,
        file_name=f"{selected_stock}_historical_data.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>📈 AI-Powered Stock Forecasting Dashboard</h3>
    <p>Hybrid CNN + LSTM Deep Learning Model</p>
    <p>Built with ❤️ using Python, TensorFlow, Streamlit, Plotly &amp; Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)