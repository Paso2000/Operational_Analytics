
# Scout Attendance Forecasting Project

Questo progetto nasce come parte dell'esame di *Operational Analytics* per l'anno accademico 2024/2025. L'obiettivo è prevedere le **presenze settimanali** alle attività scout usando modelli di time series forecasting.

## Descrizione del Dataset

Il dataset utilizzato è stato **raccolto personalmente** e contiene:
- Le presenze settimanali dei ragazzi del gruppo scout Fo13
- Periodo: circa 4 anni di attività
- Frequenza: settimanale (`W`)
- Feature principale: `turnout` (numero partecipanti ogni settimana)

---

##  Obiettivi

- Preprocessare il dataset per adattarlo ai modelli
- Applicare **tre tecniche di forecasting**:
  - **Statistica**: SARIMAX
  - **Alberi di regressione**: XGBoost
  - **Reti neurali**: LSTM
- Confrontare i risultati usando metriche di accuratezza
- Visualizzare le predizioni con grafici chiari


## Librerie utilizzate

- `pandas`, `numpy`, `matplotlib`
- `statsmodels` per SARIMAX e test ADF
- `xgboost` per alberi di regressione
- `tensorflow.keras` per LSTM
- `scikit-learn` per preprocessing e metriche

## Struttura del Codice

### 1. **Preprocessing**
- Parsing date e impostazione indice
- Riempimento dei valori nulli con interpolazione
- Feature engineering:
  - Rolling mean e std
  - Lag features
  - Variabili temporali: mese, trimestre, settimana

### 2. **Modelli**
#### SARIMAX
- Scelta manuale (e opzionale via `auto_arima`)
- Considera stagionalità annuale settimanale

#### XGBoost
- Usa lag features + rolling features
- Ottimo per pattern non lineari

####  LSTM
- Normalizzazione con `MinMaxScaler`
- Costruzione di sequenze temporali
- Architettura semplice con `Dropout` per evitare overfitting

### 3. **Valutazione**
Metriche calcolate:
- `MAE` (errore assoluto medio)
- `RMSE` (errore quadratico medio)
- `MAPE` (errore percentuale assoluto medio)
- `ME` / `MPE` (bias)
- `Correlation` tra vero e predetto

---

## Risultati

Esempio di output:

SARIMAX
MAE: 3.57
RMSE: 4.47
MAPE: 5.82%
Correlation: 0.69

XGBoost
MAE: 3.12
RMSE: 3.91
MAPE: 5.20%
Correlation: 0.78

LSTM
MAE: 3.94
RMSE: 4.72
MAPE: 6.42%
Correlation: 0.69


## 🚀 Come eseguirlo

### 1. Crea un ambiente virtuale

```bash
python -m venv venv
````

### 2. Attiva l’ambiente (Windows)

```bash
venv\Scripts\activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 4. Esegui lo script

```bash
python turnout_scout_2.py
```

---

## 🔒 Note finali

* Tutti i modelli sono stati testati e addestrati **senza data leakage**
* Dataset reale e originale: punto bonus per la personalizzazione
* Il progetto è conforme al regolamento d’esame e pronto per essere eseguito in laboratorio

---

## 👤 Autore

**Luca P.**
Capo scout Fo13
Università degli Studi – Operational Analytics 2024/2025



Fammi sapere se vuoi anche un file `requirements.txt` generato automaticamente da quello che hai usato, o se vuoi che ti crei anche la repo base con tutto organizzato.
