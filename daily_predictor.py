import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import os

# === CONFIG ===
TICKER = "XEG.TO"
MODEL_PATH = f"{TICKER}_rf_model.joblib"

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")

# === DATA FETCHING ===
df = yf.download(TICKER, period="90d")
df["Return"] = df["Close"].pct_change()
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["RSI"] = 100 - (100 / (1 + df["Close"].diff().clip(lower=0).rolling(14).mean() /
                            -df["Close"].diff().clip(upper=0).rolling(14).mean()))
df.dropna(inplace=True)

features = ["Return", "SMA_10", "SMA_50", "RSI"]
latest = df[features].iloc[[-1]]

# === LOAD MODEL & PREDICT ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
signal = model.predict(latest)[0]
label = "🟢 BUY" if signal == 1 else "🔴 SELL"

# === EMAIL SETUP ===
message = f"📊 Signal du jour pour {TICKER} : {label}\n\nDate : {datetime.today().date()}"
msg = MIMEText(message)
msg["Subject"] = f"Signal ML - {TICKER} ({datetime.today().date()})"
msg["From"] = EMAIL_SENDER
msg["To"] = EMAIL_RECIPIENT

# === SEND EMAIL ===
try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
    print("✅ Email envoyé avec succès.")
except Exception as e:
    print("❌ Échec de l'envoi de l'email :", e)
