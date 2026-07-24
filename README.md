# 📈 AI-Powered Stock Forecasting Dashboard

A professional stock market forecasting dashboard built using **Python, Streamlit, TensorFlow, and Plotly**. The application uses a **Hybrid CNN + LSTM Deep Learning Model** to predict the next day's stock closing price and provides interactive visualizations with technical indicators.
https://ai-stock-forecasting-dashboard-8p9puxrrc6st8q8ffvydiu.streamlit.app/

---

## 🚀 Features

- 📊 Interactive Stock Dashboard
- 🤖 AI-Based Next-Day Stock Price Prediction
- 📈 Historical Price Visualization
- 📉 Moving Averages (SMA 20 & SMA 50)
- 📦 Volume Analysis
- 📄 Historical Data Table
- 🎨 Professional Dark Theme
- ⚡ Real-Time Data from Yahoo Finance

---

## 🛠️ Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pandas
- Plotly
- Scikit-learn
- Yahoo Finance API
- Joblib

---

## 📂 Project Structure

```text
Stock_Forecasting/
│
├── app.py
├── train_model.py
├── utils.py
├── requirements.txt
├── README.md
│
├── assets/
│   ├── logo.png
│   └── style.css
│
├── models/
│   ├── cnn_lstm_model.keras
│   └── scaler.pkl
│
├── data/
│   └── stocks/
│
└── notebooks/
    └── training.ipynb
```

---

## 📊 Model Architecture

Hybrid CNN + LSTM

- Conv1D
- MaxPooling1D
- LSTM
- Dropout
- Dense

---

## 📈 Workflow

1. Download historical stock data
2. Preprocess and normalize data
3. Create 60-day sequences
4. Train the Hybrid CNN + LSTM model
5. Predict the next day's closing price
6. Visualize predictions in the Streamlit dashboard

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Stock-Forecasting-Dashboard.git
```

Go to the project folder:

```bash
cd AI-Stock-Forecasting-Dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- 7-Day Stock Price Forecast
- Buy / Hold / Sell Recommendation
- Candlestick Charts
- MACD & Bollinger Bands
- Sentiment Analysis using News Headlines
- Portfolio Tracking

---

## 👨‍💻 Author

Taniprava Sahoo

B.Tech CSE (AI & ML)

GitHub: https://github.com/TPS1234795

LinkedIn: https://www.linkedin.com/in/taniprava-sahoo/
