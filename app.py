
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib
import os

# === PAGE SETUP ===
st.set_page_config(page_title="ML Trading App", layout="wide")
st.title("📈 Machine Learning Trading Strategy")

# === SIDEBAR ===
etf_list = ["XEG.TO","XIT.TO","XGD.TO"]
ticker = st.sidebar.selectbox("🔍 Choisis un ETF", etf_list)
start_date = st.sidebar.date_input("📅 Date de début", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("📅 Date de fin", datetime.today())

retrain_mode = st.sidebar.radio(
    "🔄 Choix du modèle",
    ("🔁 Réentrainer automatiquement", "📦 Charger modèle sauvegardé si dispo")
)

if start_date >= end_date:
    st.sidebar.error("La date de début doit être antérieure à la date de fin.")
    st.stop()

# === DATA LOADING ===
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error("Aucune donnée téléchargée. Vérifie le ticker.")
    st.stop()

df = df[["Close"]].copy()
df["Return"] = df["Close"].pct_change()
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["RSI"] = 100 - (100 / (1 + df["Close"].diff().clip(lower=0).rolling(14).mean() /
                               -df["Close"].diff().clip(upper=0).rolling(14).mean()))
df["Future Return"] = df["Close"].shift(-1) / df["Close"] - 1
df.dropna(inplace=True)

df["Target"] = np.where(df["Future Return"] > 0, 1, 0)
features = ["Return", "SMA_10", "SMA_50", "RSI"]
X = df[features]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model_filename = f"{ticker}_rf_model.joblib"

# === MODEL TRAINING/LOADING ===
if retrain_mode == "📦 Charger modèle sauvegardé si dispo" and os.path.exists(model_filename):
    model = joblib.load(model_filename)
    st.sidebar.success("Modèle chargé avec succès.")
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)
    st.sidebar.info("Modèle entraîné et sauvegardé.")

df["Prediction"] = model.predict(X)

# === STRATEGY SIMULATION ===
initial_cash = 10000.0
cash, shares = initial_cash, 0.0
portfolio, buy_signals, sell_signals = [], [], []
cooldown_period, last_action = 5, -5

for i in range(len(df)):
    date = df.index[i]
    price = float(df["Close"].iloc[i])
    signal = df["Prediction"].iloc[i]

    if signal == 1 and cash > 0 and (i - last_action > cooldown_period):
        shares = cash / price
        cash = 0.0
        buy_signals.append((date, price))
        last_action = i
    elif signal == 0 and shares > 0 and (i - last_action > cooldown_period):
        cash = shares * price
        shares = 0.0
        sell_signals.append((date, price))
        last_action = i
    portfolio.append(cash + shares * price)

df["ML Strategy Value"] = portfolio
df["BuyHold Value"] = initial_cash * (df["Close"] / df["Close"].iloc[0])

# === PERFORMANCE METRICS ===
ml_return = df["ML Strategy Value"].iloc[-1] - initial_cash
buyhold_return = df["BuyHold Value"].iloc[-1] - initial_cash
st.subheader("📊 Résultats")
st.metric("ML Strategy Return ($)", f"{ml_return:.2f}")
st.metric("Buy & Hold Return ($)", f"{buyhold_return:.2f}")

# === TRADE GRAPH ===
st.subheader("📉 Évolution des portefeuilles")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["ML Strategy Value"], label="ML Strategy", color="purple")
ax.plot(df.index, df["BuyHold Value"], label="Buy & Hold", linestyle="--", color="orange")

if buy_signals:
    dates, vals = zip(*[(d, df.loc[d, "ML Strategy Value"]) for d, _ in buy_signals])
    ax.scatter(dates, vals, color='green', marker='^', label="Buy", s=80)
    for d in dates:
        ax.axvline(d, color='green', linestyle=':', alpha=0.2)

if sell_signals:
    dates, vals = zip(*[(d, df.loc[d, "ML Strategy Value"]) for d, _ in sell_signals])
    ax.scatter(dates, vals, color='red', marker='v', label="Sell", s=80)
    for d in dates:
        ax.axvline(d, color='red', linestyle=':', alpha=0.2)

ax.set_xlabel("Date")
ax.set_ylabel("Valeur ($)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

trades_df = pd.DataFrame(columns=["Buy Date", "Buy Price", "Sell Date", "Sell Price"])
for buy, sell in zip(buy_signals, sell_signals):
    trades_df = pd.concat([trades_df, pd.DataFrame([{
        "Buy Date": buy[0], "Buy Price": buy[1],
        "Sell Date": sell[0], "Sell Price": sell[1]
    }])], ignore_index=True)

trades_df["Gain ($)"] = trades_df["Sell Price"] - trades_df["Buy Price"]
trades_df["Durée (jours)"] = pd.to_datetime(trades_df["Sell Date"]) - pd.to_datetime(trades_df["Buy Date"])

if st.button("📁 Exporter les résultats en Excel"):
    excel_name = f"{ticker}_ml_strategy_results.xlsx"
    with pd.ExcelWriter(excel_name, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Données complètes")
        trades_df.to_excel(writer, sheet_name="Signaux", index=False)
        pd.DataFrame({
            "Stratégie": ["ML", "Buy & Hold"],
            "Rendement ($)": [ml_return, buyhold_return]
        }).to_excel(writer, sheet_name="Résumé", index=False)
    st.success(f"Fichier exporté : {excel_name}")

st.subheader("🔮 Signal prédit pour demain")
latest = df[features].iloc[[-1]]
signal = model.predict(latest)[0]
prediction_label = "🟢 BUY" if signal == 1 else "🔴 SELL"
st.markdown(f"**Signal prédictif**: {prediction_label}")

log_path = f"{ticker}_prediction_log.csv"
new_row = pd.DataFrame([{
    "Date": datetime.today().date(),
    "Signal": "BUY" if signal == 1 else "SELL"
}])
if os.path.exists(log_path):
    prediction_log = pd.read_csv(log_path)
    if str(datetime.today().date()) not in prediction_log["Date"].astype(str).values:
        prediction_log = pd.concat([prediction_log, new_row], ignore_index=True)
        prediction_log.to_csv(log_path, index=False)
else:
    prediction_log = new_row
    prediction_log.to_csv(log_path, index=False)

st.subheader("🗂️ Historique des signaux journaliers")
st.dataframe(prediction_log)

st.subheader("📋 Statistiques des trades")
st.dataframe(trades_df)
st.markdown("**📈 Statistiques Résumées :**")
st.write(f"- Nombre de trades: {len(trades_df)}")
st.write(f"- Rendement moyen ($): {trades_df['Gain ($)'].mean():.2f}")
st.write(f"- Durée moyenne d’un trade: {trades_df['Durée (jours)'].mean()}")
st.write(f"- Trades gagnants: {(trades_df['Gain ($)'] > 0).sum()} / {len(trades_df)}")
