import os
import pathlib
import json
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

# Optional imports
try:
    import joblib
except Exception:
    joblib = None

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import ta as ta
except Exception:
    yf = None

# Optional XGBoost
XGB_AVAILABLE = True
try:
    from xgboost import XGBClassifier
except Exception:
    XGB_AVAILABLE = False

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


os.environ["STREAMLIT_GLOBAL_DATA_FOLDER"] = "/tmp/.streamlit"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false" 
os.makedirs("/tmp/.streamlit", exist_ok=True)


st.set_page_config(page_title="Stock Movement Predictor (Up/Sideways/Down)", layout="wide")
st.title("📈 Stock Movement Predictor — Classical ML (Streamlit)")
st.caption("ITI105 project demo • Predict next-day **Up / Sideways / Down** using technical indicators and classical ML models.")
st.caption("Author: Isak Rabin (4466624P), Woo Ka Keung Alex (8148468Y)")

DEFAULT_THRESHOLD_PCT = 0.3

# ----------------------------
# Feature engineering (manual)
# ----------------------------
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        if col not in out.columns:
            raise ValueError(f"Missing column '{col}' in dataframe")
    if "Volume" not in out.columns:
        out["Volume"] = np.nan

    # MAs & EMAs
    out["MA5"] = out["Close"].rolling(window=5).mean()
    out["MA10"] = out["Close"].rolling(window=10).mean()
    out["MA20"] = out["Close"].rolling(window=20).mean()
    out["EMA10"] = ta.trend.EMAIndicator(close=out["Close"], window=10).ema_indicator()
    out["EMA20"] = ta.trend.EMAIndicator(close=out["Close"], window=20).ema_indicator()

    # RSI(14)
    out["RSI"] = ta.momentum.RSIIndicator(close=out["Close"], window=14).rsi()

    # MACD
    macd_indicator = ta.trend.MACD(close=out["Close"])
    out["MACD"] = macd_indicator.macd()
    out["MACD_SIGNAL"] = macd_indicator.macd_signal()

    # Bollinger Band width (20, 2)
    bb_indicator = ta.volatility.BollingerBands(close=out["Close"], window=20)
    out["BB_width"] = bb_indicator.bollinger_wband()

    # ATR(14)
    atr_indicator = ta.volatility.AverageTrueRange(
        high=out["High"],
        low=out["Low"],
        close=out["Close"],
        window=14
    )
    out["ATR"] = atr_indicator.average_true_range()

    # Stochastic %K (14)
    stoch_k = ta.momentum.StochasticOscillator(
        high=out["High"],
        low=out["Low"],
        close=out["Close"],
        window=14,
        smooth_window=3
    )
    out["Stoch_K"] = stoch_k.stoch()
    out["Stoch_D"] = stoch_k.stoch_signal()

    # Extras
    out["LOG_VOLUME"] = np.log1p(out["Volume"].dropna())
    out["MA_VOLUME_20"] = out["Volume"].rolling(window=20).mean()
    out["DAILY_RANGE"] = out["High"] - out["Low"]
    out["CLOSE_TO_OPEN_GAP"] = (out["Open"] - out["Close"].shift(1)) / out["Close"].shift(1)
    return out

def compute_next_day_pct_change(df: pd.DataFrame) -> pd.Series:
    # % change from t to t+1, in PERCENT units, aligned to day t
    pct = df["Close"].pct_change(fill_method=None).mul(100)
    return pct.shift(-1)

def make_labels(pct, threshold):
    if pd.isna(pct):
        return np.nan
    if pct >= threshold:
        return "Up"
    elif pct <= -threshold:
        return "Down"
    else:
        return "Sideway" 

FEATURE_COLS_DEFAULT = [
    "Open","High","Low","Close",
    "MA20","RSI","MACD","EMA20","BB_WIDTH","ATR","STOCH_K",
    "LOG_VOLUME", "MA_VOLUME_20","DAILY_RANGE","CLOSE_TO_OPEN_GAP"
]

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_yf(ticker: str, start: date, end: date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed. Add 'yfinance' to requirements.txt")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).title() for c in df.columns]
    return df

def prepare_features(df: pd.DataFrame, thr_pct: float, feature_cols: list | None):
    df = df.copy()
    df.columns = [c.strip().title() for c in df.columns]
    req = ["Date","Open","High","Low","Close"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    feats = compute_technical_indicators(df)
    feats["Pct_Change"] = compute_next_day_pct_change(feats)
    feats["Label"] = feats["Pct_Change"].apply(make_labels, args=(thr_pct,))

    feats = feats.dropna().reset_index(drop=True)

    cols = feature_cols if feature_cols else [c for c in FEATURE_COLS_DEFAULT if c in feats.columns]
    X = feats[cols].copy()
    y = feats["Label"].copy()
    return X, y, feats, cols

# ----------------------------
# Artifact loading & training
# ----------------------------
def load_artifact_for(model_key: str, artifacts_dir: str):
    names = {
        "logreg": "logreg",
        "dt": "decision_tree",
        "rf": "random_forest",
        "xgb": "xgboost",
    }
    model_path = os.path.join(artifacts_dir, f"{names[model_key]}.joblib")
    feature_json = os.path.join(artifacts_dir, f"{names[model_key]}.json")
    model = None
    feat_cols = None

    if model_key == "xgb" and not os.path.exists(model_path):
        alt = os.path.join(artifacts_dir, "xgboost.json")
        if os.path.exists(alt) and XGB_AVAILABLE:
            m = XGBClassifier()
            m.load_model(alt)
            model = m

    if model is None and joblib is not None and os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.warning(f"Could not load {model_path}: {e}")

    if os.path.exists(feature_json):
        try:
            with open(feature_json, "r") as f:
                meta = json.load(f)
            feat_cols = meta.get("feature_columns")
        except Exception as e:
            st.warning(f"Could not parse {names[model_key]}.json: {e}")
    return model, feat_cols

def train_quick_model(model_key: str, X: pd.DataFrame, y: pd.Series):
    if model_key == "logreg":
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)),
                         ("clf", LogisticRegression(max_iter=1000, multi_class="auto"))])
        pipe.fit(X, y)
        return pipe
    elif model_key == "dt":
        clf = DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=20)
        clf.fit(X, y)
        return clf
    elif model_key == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X, y)
        return clf
    elif model_key == "xgb":
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost not installed. Add 'xgboost' to requirements.txt.")
        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
            eval_metric="mlogloss", tree_method="hist"
        )
        clf.fit(X.values, y.values)
        return clf
    else:
        raise ValueError("Unknown model key")

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="^GSPC")
    ds = st.date_input("Start date", value=date(2015,1,1))
    de = st.date_input("End date", value=date.today())
    thr_pct = st.number_input("Movement threshold ±% (daily close)", min_value=0.1, max_value=2.0, step=0.1, value=DEFAULT_THRESHOLD_PCT)
    st.markdown("---")
    st.subheader("Choose model")
    model_name = st.selectbox("Model", ["Logistic Regression","Decision Tree","Random Forest","XGBoost (CPU)"])
    model_key = {"Logistic Regression":"logreg","Decision Tree":"dt","Random Forest":"rf","XGBoost (CPU)":"xgb"}[model_name]

    st.markdown("---")
    st.subheader("Data source")
    use_csv = st.checkbox("Use uploaded CSV instead of Yahoo Finance", value=False)
    up = st.file_uploader("Upload CSV with columns: Date, Open, High, Low, Close, Volume", type=["csv"]) if use_csv else None

    st.markdown("---")
    artifacts_dir = st.text_input("Artifacts directory", value="artifacts", help="Place your pre-trained models here")
    st.caption("Expected files (optional): logreg.joblib, decision_tree.joblib, random_forest.joblib, xgboost.joblib/xgboost.json, feature_columns.json")
    run_btn = st.button("Run")

st.markdown("> **Tip**: If a model artifact isn't found for your selection, the app will train a quick demo model on the downloaded data.")

if not run_btn:
    st.stop()

# Load data
try:
    if use_csv and up is not None:
        raw = pd.read_csv(up)
        raw.columns = [c.strip().title() for c in raw.columns]
        if "Date" in raw.columns:
            raw["Date"] = pd.to_datetime(raw["Date"])
    else:
        if yf is None:
            st.error("yfinance is not installed. Please add it to requirements.txt.")
            st.stop()
        raw = load_yf(ticker, ds, de)
    if raw is None or raw.empty:
        st.warning("No data loaded. Check ticker/dates or upload a valid CSV.")
        st.stop()
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# Prepare features
try:
    artifact_model, artifact_feats = load_artifact_for(model_key, artifacts_dir)
    X, y, feats, cols = prepare_features(raw, thr_pct, artifact_feats)
except Exception as e:
    st.error(f"Feature preparation error: {e}")
    st.stop()

# Align columns for artifact model
if artifact_model is not None:
    try:
        if artifact_feats:
            missing = [c for c in artifact_feats if c not in X.columns]
            if missing:
                st.error(f"The current dataset is missing some features expected by the artifact model: {missing}")
                st.stop()
            X_use = X[artifact_feats]
        else:
            X_use = X
    except Exception as e:
        st.error(f"Error aligning features with artifact model: {e}")
        st.stop()
    model = artifact_model
    model_src = "artifact"
else:
    with st.spinner("Training a quick demo model (no artifact found)..."):
        try:
            model = train_quick_model(model_key, X, y)
            X_use = X
            model_src = "quick-train"
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

# Predict last N rows
N = 30
try:
    y_pred = model.predict(X_use)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

out = pd.DataFrame(index=feats.index)
out["Date"] = feats["Date"].values if "Date" in feats.columns else np.arange(len(feats))
out["Close"] = feats["Close"].values
out["TrueLabel"] = feats["Label"].values
out["Predicted"] = y_pred

# probs (if available)
proba = None
classes = None
if hasattr(model, "predict_proba"):
    try:
        proba = model.predict_proba(X_use)
        classes = getattr(model, "classes_", None)
        if classes is not None and proba is not None:
            proba_df = pd.DataFrame(proba, columns=[f"P({c})" for c in classes])
            out = pd.concat([out, proba_df], axis=1)
    except Exception:
        proba = None

# Show results
c1, c2 = st.columns([1.2, 1])
with c1:
    st.subheader("Recent predictions")
    st.dataframe(out.tail(N), use_container_width=True, hide_index=True)
    st.caption(f"Model source: **{model_src}** • Features used: {len(cols)}")

with c2:
    st.subheader("Close price (last 200)")
    last200 = feats[["Date","Close"]].tail(200).set_index("Date")
    st.line_chart(last200)
    st.subheader("Latest prediction")
    last_row = out.iloc[-1]
    st.metric("Predicted next-day movement", str(last_row["Predicted"]))

    if proba is not None and classes is not None:
        try:
            p = proba[-1]
            prob_frame = pd.DataFrame({"Class": classes, "Probability": p}).set_index("Class")
            st.bar_chart(prob_frame)
        except Exception:
            pass

with st.expander("Details"):
    st.write("**Ticker:**", ticker)
    st.write("**Date range:**", ds, "to", de)
    st.write("**Threshold (±%)**:", thr_pct)
    st.write("**Feature columns:**", cols)
    st.write("**Rows used (after indicator warm-up):**", len(feats))

st.success("Done. To use your exact trained models, upload them into the `artifacts/` folder with the expected filenames.")

