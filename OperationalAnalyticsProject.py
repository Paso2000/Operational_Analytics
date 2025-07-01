
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from dm_test import dm_test
# -------------------------------------
# Funzione per creare dataset per modelli LSTM o simili
# Trasforma una serie temporale in un dataset supervisionato
# look_back = numero di valori passati usati come input
# -------------------------------------
def create_dataset(data, look_back=52):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

# -------------------------------------
# Funzione di valutazione della qualità del forecast
# Calcola: MAE, RMSE, MAPE, ME, MPE, Correlazione
# -------------------------------------
def evaluate(true, pred):
    true = np.array(true).ravel()
    pred = np.array(pred).ravel()
    
    mape = np.mean(np.abs(pred - true) / np.abs(true)) * 100  # Mean Absolute Percentage Error
    mae = np.mean(np.abs(pred - true))                        # Mean Absolute Error
    rmse = np.sqrt(np.mean((pred - true)**2))                 # Root Mean Squared Error
    me = np.mean(pred - true)                                 # Mean Error
    mpe = np.mean((pred - true) / true) * 100                 # Mean Percentage Error
    corr = np.corrcoef(true, pred)[0, 1]                      # Pearson correlation coefficient

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'ME': me,
        'MPE': mpe,
        'Correlation': corr
    }

# -------------------------------------
# CARICAMENTO E PREPROCESSING DEI DATI
# -------------------------------------

# Carica il file CSV
df = pd.read_csv("turnout_gruppo_Fo13.csv")

# Converte la colonna 'date' in datetime
df["date"] = pd.to_datetime(df["date"])

# Imposta la colonna data come indice temporale
df.set_index("date", inplace=True)

# Forza frequenza settimanale e interpola i valori mancanti
df = df.asfreq("W")
df["turnout"].interpolate(inplace=True)

# -------------------------------------
# ANALISI ACF E PACF
# -------------------------------------

# Autocorrelazione per determinare relazioni tra periodi passati
plot_acf(df["turnout"], lags=40)
plt.title("Autocorrelation (ACF)")
plt.show()

# Autocorrelazione parziale per identificare l’ordine AR
plot_pacf(df["turnout"], lags=40, method='ywm')
plt.title("Partial Autocorrelation (PACF)")
plt.show()

# -------------------------------------
# TEST DI STAZIONARIETÀ (ADF Test)
# -------------------------------------

# Verifica se la serie è stazionaria
adf_result = adfuller(df["turnout"])

print("\n=== Risultati ADF Test ===")
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"P-value: {adf_result[1]:.4f}")
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value:.4f}")

#Il p-value < 0.05, la serie è stazionaria e non richiede differenziazione

# Divido i dati in train e test, come ptest prendo l'ultimo anno(52 settimane)
train = df.iloc[:-52]
test = df.iloc[-52:]

#----algoritmo per cercare i migliori valori per order e seasonal order
# model = pm.auto_arima(df["turnout"], start_p=1, start_q=1,
# test='adf', max_p=3, max_q=3, m=4,
# start_P=0, seasonal=True,
# d=None, D=1, trace=True,
# error_action='ignore',
# suppress_warnings=True,
# stepwise=True) # stepwise=False full grid
# print(model.summary())

# Stampa il modello migliore trovato
#print(auto_model.summary())

# -------------------------------------
# 1. MODELLO SARIMA (Statistical)
# -------------------------------------

# Creazione e training del modello SARIMA
SARIMA_model = SARIMAX(
    train["turnout"],
    order=(1, 1, 1), #model.order              # Ordine ARIMA (p,d,q)
    seasonal_order=(1, 0, 1, 52)) #model.seasonal_order  # Ordine stagionale (P,D,Q,s) con stagionalità settimanale
SARIMA_trained = SARIMA_model.fit()

# Forecast sul periodo di test
forecast_sarima = SARIMA_trained.forecast(steps=len(test))

# Valutazione del modello
sarima_metrics = evaluate(test["turnout"], forecast_sarima)

# Plot dei risultati
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["turnout"], label="True")
plt.plot(test.index, forecast_sarima, label="SARIMA")
plt.legend()
plt.title("Turnout Forecast SARIMA Result")
plt.show()
SARIMA_trained.plot_diagnostics(figsize=(12,6))
plt.show()


# -------------------------------------
# 2. MODELLO XGBoost (Regression Trees)
# -------------------------------------

# Copia del dataframe per costruire le feature
df_feat = df.copy()

# Feature lag (valori delle settimane precedenti)
for lag in range(1, 53):
    df_feat[f"lag_{lag}"] = df_feat["turnout"].shift(lag)


# Estrai informazioni temporali come feature
df_feat["month"] = df_feat.index.month
df_feat["weekofyear"] = df_feat.index.isocalendar().week
df_feat["quarter"] = df_feat.index.quarter
df_feat["year"] = df_feat.index.year

# Feature statistiche mobili (rolling mean & std)
# Aggiunge media e deviazione standard mobile su 4 settimane
df_feat["rolling_mean_4"] = df_feat["turnout"].rolling(window=4).mean()
df_feat["rolling_std_4"] = df_feat["turnout"].rolling(window=4).std()

# Rimozione delle righe con NaN (dovute a lag/rolling)
df_feat.dropna(inplace=True)

# Split tra input (X) e output (y)
X = df_feat.drop("turnout", axis=1)
y = df_feat["turnout"]

# Divisione train/test basata su date
X_train, X_test = X.loc[:train.index[-1]], X.loc[test.index[0]:]
y_train, y_test = y.loc[:train.index[-1]], y.loc[test.index[0]:]

# Addestramento del modello XGBoost
xgb_model = XGBRegressor(objective='reg:squarederror',n_estimators=100)
xgb_model.fit(X_train, y_train)

# Predizione e valutazione
xgb_pred = xgb_model.predict(X_test)
xgb_metrics = evaluate(y_test, xgb_pred)

# Plot dei risultati
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["turnout"], label="True")
plt.plot(test.index, xgb_pred, label="XGBoost")
plt.legend()
plt.title("Turnout Forecast XGBoost Result")
plt.show()


# -------------------------------------
# 3. MODELLO LSTM (Neural Network)
# -------------------------------------

# Normalizzazione dei dati per LSTM (range 0-1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["turnout"]])

# Creazione sequenze temporali (X: input, y: target)
X_lstm, y_lstm = create_dataset(scaled)

# Divisione train/test sugli ultimi 52 punti
X_lstm_train, X_lstm_test = X_lstm[:-52], X_lstm[-52:]
y_lstm_train, y_lstm_test = y_lstm[:-52], y_lstm[-52:]

# Costruzione rete neurale LSTM
lstm_model = Sequential([
    # Primo layer LSTM con return_sequences per il secondo LSTM
    LSTM(64, return_sequences=True, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])),
    Dropout(0.2),
    # Secondo layer LSTM (non ritorna sequenze)
    LSTM(32),
    Dropout(0.2),
    # Output layer (regressione su un valore)
    Dense(1)
])

# Compilazione
lstm_model.compile(optimizer="adam", loss="mse")

# Addestramento modello
lstm_model.fit(X_lstm_train, y_lstm_train, epochs=60, batch_size=16, verbose=0)

# Predizione e denormalizzazione
lstm_pred = lstm_model.predict(X_lstm_test)
lstm_pred_inv = scaler.inverse_transform(lstm_pred)

# Valutazione del modello
lstm_metrics = evaluate(test["turnout"].values.ravel(), lstm_pred_inv.ravel())

# Plot dei risultati
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["turnout"], label="True")
plt.plot(test.index, lstm_pred_inv, label="LSTM")
plt.legend()
plt.title("Turnout Forecast LSTM Result")
plt.show()


# -------------------------------------
# CONFRONTO FINALE DEI MODELLI
# -------------------------------------

# Funzione per stampa dei risultati
def print_metrics(name, metrics): 
    print(f"\n{name}")
    for k, v in metrics.items():
        if k in ['MAPE', 'MPE']:
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.2f}")


print("=== Turnout Forecast Metrics ===")
print_metrics("SARIMA", sarima_metrics)
print_metrics("XGBoost", xgb_metrics)
print_metrics("LSTM", lstm_metrics)

# Plot unico di confronto tra tutti i modelli
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["turnout"], label="True")
plt.plot(test.index, forecast_sarima, label="SARIMA")
plt.plot(test.index, xgb_pred, label="XGBoost")
plt.plot(test.index, lstm_pred_inv, label="LSTM")
plt.legend()
plt.title("Turnout Forecast - Confronto Finale")
plt.show()

# --- 1. SARIMA su tutta la serie ---
sarima_pred_all = SARIMA_trained.predict(start=df.index[1], end=df.index[-1])

# --- 2. XGBoost su tutto il dataset (dove possibile) ---
xgb_pred_all = xgb_model.predict(X)

# --- 3. LSTM su tutta la serie ---
X_lstm_all, _ = create_dataset(scaled)
lstm_pred_all = lstm_model.predict(X_lstm_all)
lstm_pred_all_inv = scaler.inverse_transform(lstm_pred_all)

# --- Ricostruzione indice temporale per LSTM ---
lstm_index_start = df.index[12]  # i primi 12 non esistono perché usati come lookback
lstm_index = df.index[12:]


true = test["turnout"].values
sarima_forecast = forecast_sarima
xgb_forecast = xgb_pred
lstm_forecast = lstm_pred_inv.ravel()
# Esegui i test Diebold-Mariano
dm_sarima_vs_xgb = dm_test(true, sarima_forecast, xgb_forecast, h=1, crit="MSE")
dm_sarima_vs_lstm = dm_test(true, sarima_forecast, lstm_forecast, h=1, crit="MSE")
dm_xgb_vs_lstm = dm_test(true, xgb_forecast, lstm_forecast, h=1, crit="MSE")

# Stampa i risultati
print("\n--- Diebold-Mariano Test ---")
print(f"SARIMA vs XGBoost:     DM = {dm_sarima_vs_xgb.DM:.3f}, p-value = {dm_sarima_vs_xgb.p_value:.4f}")
print(f"SARIMA vs LSTM:        DM = {dm_sarima_vs_lstm.DM:.3f}, p-value = {dm_sarima_vs_lstm.p_value:.4f}")
print(f"XGBoost vs LSTM:       DM = {dm_xgb_vs_lstm.DM:.3f}, p-value = {dm_xgb_vs_lstm.p_value:.4f}")

# --- PLOT COMPLETO ---
plt.figure(figsize=(14,6))
plt.plot(df.index, df["turnout"], label="True", color='black')
plt.plot(sarima_pred_all.index, sarima_pred_all, label="SARIMA", linestyle='--')
plt.plot(X.index, xgb_pred_all, label="XGBoost", linestyle='--')
plt.plot(lstm_index, lstm_pred_all_inv.ravel(), label="LSTM", linestyle='--')
plt.title("Predizioni di tutti i modelli sull'intera serie temporale")
plt.xlabel("Data")
plt.ylabel("Turnout")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



