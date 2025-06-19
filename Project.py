
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings("ignore")

def create_dataset(data, look_back=12):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

# Load and preprocess
df = pd.read_csv("turnout_gruppo_Fo13.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df = df.asfreq("W")
df["turnout"].interpolate(inplace=True)

# --- ADF Test ---
adf_result = adfuller(df["turnout"])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value}")

# --- ACF and PACF Plots ---
plot_acf(df["turnout"], lags=40)
plt.title("Autocorrelation (ACF)")
plt.show()

plot_pacf(df["turnout"], lags=40, method='ywm')
plt.title("Partial Autocorrelation (PACF)")
plt.show()

df["month"] = df.index.month
df["weekofyear"] = df.index.isocalendar().week
df["quarter"] = df.index.quarter
df["year"] = df.index.year


df["rolling_mean_4"] = df["turnout"].rolling(window=4).mean()
df["rolling_std_4"] = df["turnout"].rolling(window=4).std()

# Train/test split
train = df.iloc[:-52]
test = df.iloc[-52:]

#----algoritmo per cercare i migliori valori per order e seasonal order
# auto_model = pm.auto_arima(
#     train["turnout"],
#     seasonal=True,
#     m=52,  # stagionalità annuale settimanale
#     stepwise=True,
#     suppress_warnings=True,
#     trace=True  # mostra i tentativi
# )

# Stampa il modello migliore trovato
#print(auto_model.summary())

# 1. SARIMAX

SARIMAX_model = SARIMAX(
    train["turnout"],
    order=(1,1,1),
    seasonal_order=(1,0,1,52)
)
SARIMAX_result = SARIMAX_model.fit()
forecast_sarimax = SARIMAX_result.forecast(steps=len(test))


# 2. XGBoost
df_feat = df.copy()

# Lag feature
for lag in range(1, 13):
    df_feat[f"lag_{lag}"] = df_feat["turnout"].shift(lag)

# Rolling window features
df_feat["rolling_mean_4"] = df_feat["turnout"].rolling(window=4).mean()
df_feat["rolling_std_4"] = df_feat["turnout"].rolling(window=4).std()

# Drop NA dopo lag e rolling
df_feat.dropna(inplace=True)

# Input/Output split
X = df_feat.drop("turnout", axis=1)
y = df_feat["turnout"]

X = df_feat.drop("turnout", axis=1)
y = df_feat["turnout"]
X_train, X_test = X.loc[:train.index[-1]], X.loc[test.index[0]:]
y_train, y_test = y.loc[:train.index[-1]], y.loc[test.index[0]:]

xgb_model = XGBRegressor(n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# -----------------------------
# 3. LSTM
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["turnout"]])

X_lstm, y_lstm = create_dataset(scaled)
X_lstm_train, X_lstm_test = X_lstm[:-52], X_lstm[-52:]
y_lstm_train, y_lstm_test = y_lstm[:-52], y_lstm[-52:]

lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_lstm_train, y_lstm_train, epochs=30, batch_size=16, verbose=0)

lstm_pred = lstm_model.predict(X_lstm_test)
lstm_pred_inv = scaler.inverse_transform(lstm_pred)


#Evaluate
def evaluate(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mae, rmse, mape

sarima_mae, sarima_rmse, sarima_mape = evaluate(test["turnout"], forecast_sarimax)
xgb_mae, xgb_rmse, xgb_mape = evaluate(y_test, xgb_pred)
lstm_mae, lstm_rmse, lstm_mape = evaluate(test["turnout"].values, lstm_pred_inv)

print("=== Turnout Forecast Result ===")
print(f"SARIMAX -> MAE: {sarima_mae:.2f}, RMSE: {sarima_rmse:.2f}, MAPE: {sarima_mape:.2f}%")
print(f"XGBoost -> MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}, MAPE: {xgb_mape:.2f}%")
print(f"LSTM -> MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}, MAPE: {lstm_mape:.2f}%")

# Plot 
plt.figure(figsize=(12,6))
plt.plot(test.index, test["turnout"], label="True")
plt.plot(test.index, forecast_sarimax, label="SARIMAX")
plt.plot(test.index, xgb_pred, label="XGBoost")
plt.plot(test.index, lstm_pred_inv, label="LSTM")
plt.legend()
plt.title("Turnout Forecast Result")
plt.show()
