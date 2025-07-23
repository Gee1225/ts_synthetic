# --- Core Libraries ---
import warnings
import numpy as np
import pandas as pd
import streamlit as st

# --- Machine Learning & Preprocessing ---
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# --- Deep Learning (Keras) ---
from keras.models import Sequential  # type: ignore
from keras.layers import LSTM, Dense

# --- Classical Time Series ---
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- PyTorch & Lightning Compatibility ---
import torch
from torch import nn
from torch.utils.data import Dataset
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

# --- PyTorch Forecasting Models ---
from pytorch_forecasting.models.nbeats import NBeats
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE, MAE

# --- Transformers & HF Datasets (guarded import) ---
try:
    from transformers import PatchTSTConfig, PatchTSForTimeSeriesForecasting, PatchTSTFeatureExtractor
    from datasets import Dataset as HFDataset
    HAVE_PATCHTST = True
except ImportError:
    HAVE_PATCHTST = False

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Advanced Time Series Forecasting - Only Prints MAE Based on Preset Paramenters", layout="wide")
st.title("🔮 Advanced Forecasting Model")

# ---------- Data Loading ----------
DATA_PATH = "./data"
@st.cache_data
def load_data():
    volume = pd.read_csv(f"{DATA_PATH}/volume_log.csv").assign(date=lambda df: pd.to_datetime(df['date']))
    calendar = pd.read_csv(f"{DATA_PATH}/calendar.csv").assign(date=lambda df: pd.to_datetime(df['date']))
    product = pd.read_csv(f"{DATA_PATH}/product_catalog.csv")
    weather = pd.read_csv(f"{DATA_PATH}/weather.csv").assign(date=lambda df: pd.to_datetime(df['date']))
    carrier = pd.read_csv(f"{DATA_PATH}/carrier_tracking.csv").assign(ETA=lambda df: pd.to_datetime(df['ETA']))
    order = pd.read_csv(f"{DATA_PATH}/order_feed_stream.csv").assign(order_time=lambda df: pd.to_datetime(df['order_time']))
    inventory = pd.read_csv(f"{DATA_PATH}/inventory_snapshot.csv")
    return volume, calendar, product, weather, carrier, order, inventory

volume, calendar, product, weather, carrier, order, inventory = load_data()

# ---------- XGBoost Prep ----------
@st.cache_data
def prepare_xgb():
    order['date'] = order['order_time'].dt.date.astype('datetime64[ns]')
    order_summary = order.groupby(['date','center_id','sku_id'])['quantity'].sum()\
                        .reset_index().rename(columns={'quantity':'order_volume_last_24h'})
    carrier['date'] = carrier['ETA'].dt.date.astype('datetime64[ns]')
    carrier['late_arrival_flag'] = (carrier['carrier_delay_min'] > 0).astype(int)
    carrier_summary = carrier.groupby(['date','center_id'])[['carrier_delay_min','late_arrival_flag']].mean().reset_index()

    X = volume.merge(calendar, on='date', how='left')\
              .merge(product, on='sku_id', how='left')\
              .merge(weather, on=['date','center_id'], how='left')\
              .merge(order_summary, on=['date','center_id','sku_id'], how='left')\
              .merge(carrier_summary, on=['date','center_id'], how='left')\
              .merge(inventory, on=['center_id','sku_id'], how='left')
    X['available_inventory'] = X['on_hand'] - X['reorder_point']
    X['order_volume_last_24h'] = X['order_volume_last_24h'].fillna(0)
    features = [
        'inbound_volume','is_weekend','is_holiday','is_promo',
        'weather_alert','temperature','order_volume_last_24h',
        'carrier_delay_min','late_arrival_flag','available_inventory','lead_time_days'
    ]
    X.dropna(subset=features, inplace=True)
    return X[features], X['outbound_volume']

# ---------- Sidebar & Selector ----------
st.sidebar.title("🛠 Select Forecasting Model")
model_option = st.sidebar.selectbox(
    "Model", ['XGBoost','Temporal Fusion Transformer','N-BEATS','PATCHTST','SARIMA','LSTM']
)

# ---------- XGBoost ----------
if model_option == 'XGBoost':
    st.subheader("📦 XGBoost Baseline Forecast")
    n_estimators = st.sidebar.slider("n_estimators", 100, 1000, step=100, value=100)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0, step=0.01, value=0.1)
    max_depth = st.sidebar.slider("max_depth", 1, 10, step=1, value=3)
    if st.sidebar.button("Run XGBoost"):
        X_model, y = prepare_xgb()
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        model.fit(X_model, y)
        preds = model.predict(X_model)
        st.success(f"✅ MAE: {mean_absolute_error(y,preds):.2f}, RMSE: {mean_squared_error(y,preds,squared=False):.2f}")

# ---------- SARIMA ----------
elif model_option == 'SARIMA':
    st.subheader("🔄 SARIMA Forecast")
    p = st.sidebar.number_input("AR (p)", 0, 5, value=1)
    d = st.sidebar.number_input("I (d)", 0, 2, value=1)
    q = st.sidebar.number_input("MA (q)", 0, 5, value=1)
    P = st.sidebar.number_input("Seasonal AR (P)", 0, 5, value=1)
    D = st.sidebar.number_input("Seasonal I (D)", 0, 2, value=1)
    Q = st.sidebar.number_input("Seasonal MA (Q)", 0, 5, value=1)
    s = st.sidebar.number_input("Seasonal period (s)", 1, 30, value=7)
    if st.sidebar.button("Run SARIMA"):
        sku_ts = volume.query("center_id=='C001' and sku_id=='SKU001'").set_index('date')
        res = SARIMAX(sku_ts['outbound_volume'], order=(p,d,q), seasonal_order=(P,D,Q,s)).fit(disp=False)
        preds = res.fittedvalues
        st.success(f"✅ SARIMA MAE: {mean_absolute_error(sku_ts['outbound_volume'][1:], preds[1:]):.2f}")

# ---------- LSTM ----------
elif model_option == 'LSTM':
    st.subheader("🤖 LSTM Forecast")
    seq_len = st.sidebar.slider("Sequence length", 10, 100, value=30)
    epochs  = st.sidebar.slider("Epochs", 1, 50, value=5)
    if st.sidebar.button("Run LSTM"):
        sku_ts = volume.query("center_id=='C001' and sku_id=='SKU001'").set_index('date')
        arr = sku_ts['outbound_volume'].values.reshape(-1,1)
        scaler = MinMaxScaler().fit(arr)
        scaled = scaler.transform(arr)
        Xs, ys = [], []
        for i in range(seq_len, len(scaled)):
            Xs.append(scaled[i-seq_len:i, 0])
            ys.append(scaled[i, 0])
        Xs = np.array(Xs).reshape(-1, seq_len, 1)
        ys = np.array(ys).reshape(-1, 1)
        model = Sequential([LSTM(50, input_shape=(seq_len,1)), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(Xs, ys, epochs=epochs, verbose=0)
        preds = scaler.inverse_transform(model.predict(Xs))
        true  = scaler.inverse_transform(ys)
        st.success(f"✅ LSTM MAE: {mean_absolute_error(true, preds):.2f}")

# ---------- Temporal Fusion Transformer ----------


elif model_option == 'Temporal Fusion Transformer':
    st.subheader("🧠 TFT Forecast")
    epochs = st.sidebar.slider("Epochs", 1, 10, value=3)
    if st.sidebar.button("Run TFT"):
        # prepare dataset
        df = (
            volume[volume['center_id']=='C001'][['date','sku_id','outbound_volume']]
            .assign(
                time_idx=lambda d: (d['date'] - d['date'].min()).dt.days,
                group=lambda d: d['sku_id'],
                log_vol=lambda d: np.log1p(d['outbound_volume'])
            )
        )
        ds = TimeSeriesDataSet(
            df, time_idx='time_idx', target='log_vol', group_ids=['group'],
            max_encoder_length=30, max_prediction_length=7,
            time_varying_unknown_reals=['log_vol']
        )
        loader = ds.to_dataloader(train=True, batch_size=64, num_workers=0)
        tft = TemporalFusionTransformer.from_dataset(ds, loss=MAE())
        trainer = Trainer(max_epochs=epochs, accelerator='auto', devices=1,
                         enable_checkpointing=False, gradient_clip_val=0.1,
                         callbacks=[EarlyStopping(monitor='train_loss', patience=3)])
        trainer.fit(model=tft, train_dataloaders=loader)
        # fix: move to CPU before numpy
        preds_all = tft.predict(loader).cpu().numpy()
        preds_1 = preds_all[:, 0]
        actuals = df['log_vol'].values[30:30 + len(preds_1)]
        mae_tft = mean_absolute_error(np.expm1(actuals), np.expm1(preds_1))
        st.info(f"⚙️ TFT MAE (h=1): {mae_tft:.2f}")

# ---------- N-BEATS ----------
elif model_option == 'N-BEATS':
    st.subheader("📐 N-BEATS Forecast")
    epochs = st.sidebar.slider("Epochs", 1, 10, value=3)
    if st.sidebar.button("Run N-BEATS"):
        # prepare dataset
        df = (
            volume[volume['center_id']=='C001'][['date','sku_id','outbound_volume']]
            .assign(
                time_idx=lambda d: (d['date'] - d['date'].min()).dt.days,
                group=lambda d: d['sku_id'],
                log_vol=lambda d: np.log1p(d['outbound_volume'])
            )
        )
        ds = TimeSeriesDataSet(
            df, time_idx='time_idx', target='log_vol', group_ids=['group'],
            max_encoder_length=30, max_prediction_length=7,
            time_varying_unknown_reals=['log_vol']
        )
        loader = ds.to_dataloader(train=True, batch_size=64, num_workers=0)
        nbeats_model = NBeats.from_dataset(ds, learning_rate=0.03, log_interval=10,
                                          log_val_interval=1, weight_decay=1e-2,
                                          widths=[32,512,512,512,32], backcast_loss_ratio=0.1,
                                          loss=MAE())
        trainer = Trainer(max_epochs=epochs, accelerator='auto', devices=1,
                         enable_checkpointing=False, gradient_clip_val=0.1,
                         callbacks=[EarlyStopping(monitor='train_loss', patience=3)])
        trainer.fit(model=nbeats_model, train_dataloaders=loader)
        preds_all = nbeats_model.predict(loader).cpu().numpy()
        preds_1 = preds_all[:, 0]
        actuals = df['log_vol'].values[30:30 + len(preds_1)]
        mae_nb = mean_absolute_error(np.expm1(actuals), np.expm1(preds_1))
        st.info(f"⚙️ N-BEATS MAE (h=1): {mae_nb:.2f}")

# ---------- PATCHTST ----------
elif model_option == 'PATCHTST':
    st.subheader("🧩 PATCHTST Forecast")
    if st.sidebar.button("Run PATCHTST"):
        if not HAVE_PATCHTST:
            st.error("PatchTST not available. Upgrade transformers & datasets.")
        else:
            df = (
                volume[volume['center_id']=='C001'][['date','sku_id','outbound_volume']]
                .assign(
                    time_idx=lambda d: (d['date'] - d['date'].min()).dt.days,
                    log_vol=lambda d: np.log1p(d['outbound_volume'])
                )
            )
            patch_df = df[['time_idx','log_vol']].rename(columns={'time_idx':'past_time_features','log_vol':'target'})
            hf_ds = HFDataset.from_pandas(patch_df)
            extractor = PatchTSTFeatureExtractor(seq_length=30, prediction_length=7)
            enc = extractor(hf_ds[:], return_tensors='pt')
            patch_model = PatchTSForTimeSeriesForecasting.from_pretrained('microsoft/patchtst-base')
            patch_model.eval()
            with torch.no_grad():
                out = patch_model(
                    past_time_features=enc['past_time_features'],
                    past_target=enc['past_target'],
                    future_time_features=enc['future_time_features']
                )
            preds = out.predictions.squeeze().cpu().numpy()
            actuals = patch_df['target'].values
            mae_pt = mean_absolute_error(actuals, preds)
            st.info(f"⚙️ PATCHTST MAE: {mae_pt:.2f}")
