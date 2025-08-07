import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
# Importa solo se TensorFlow è installato e necessario
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    st.warning("TensorFlow/Keras non trovato. Il modello LSTM non sarà disponibile.")

from statsmodels.tsa.seasonal import seasonal_decompose # Per la decomposizione
import warnings

# Ignora i FutureWarning di statsmodels per una pulizia visiva
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# --- CONFIGURAZIONE PAGINA STREAMLIT E STILI CSS PERSONALIZZATI ---
st.set_page_config(layout="wide", page_title="Demand Forecasting Dashboard", page_icon="📈")

st.markdown("""
<head>
    <!-- Importazione dei font per assicurare che siano disponibili -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<style>
    /* Stili generali per il body e il container principale di Streamlit */
    html, body {
        height: 100%;
        margin: 0;
        padding: 0;
    }

    .stApp {
        font-family: 'Inter', sans-serif !important;
        line-height: 1.7 !important;
        /* I colori di sfondo sono gestiti da config.toml per maggiore coerenza */
    }
    
    /* Intestazioni */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important; /* Manteniamo Inter per coerenza */
        color: var(--primaryColor, #4CAF50) !important; /* Colore primario dal config.toml */
    }

    /* Testo generale */
    p, .stText, .stMarkdown, label {
        color: var(--textColor, #333333) !important;
    }

    /* Stile per la riga orizzontale (generata da st.markdown("---")) */
    hr {
        border-top: 1px solid var(--secondaryBackgroundColor, #F0F2F6) !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }

    /* Stile per i bottoni */
    .stButton>button {
        background-color: var(--primaryColor, #4CAF50) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
        font-weight: bold !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2) !important;
    }
    .stButton>button:hover {
        background-color: #45A049 !important; /* Tonalità più scura al passaggio */
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }

    /* Slider */
    .stSlider .st-bd { /* Track */
        background: var(--secondaryBackgroundColor, #F0F2F6) !important;
    }
    .stSlider .st-be { /* Fill */
        background: var(--primaryColor, #4CAF50) !important;
    }
    .stSlider .st-bf { /* Thumb */
        background: var(--primaryColor, #4CAF50) !important;
        border: 2px solid var(--primaryColor, #4CAF50) !important;
    }

    /* Selectbox */
    .stSelectbox .st-bb { /* Dropdown arrow */
        color: var(--primaryColor, #4CAF50) !important;
    }
    .stSelectbox .st-cc { /* Selected value text */
        color: var(--textColor, #333333) !important;
        background-color: var(--backgroundColor, #FFFFFF) !important;
        border: 1px solid var(--secondaryBackgroundColor, #F0F2F6) !important;
        border-radius: 5px !important;
    }
    .stSelectbox .st-cd { /* Dropdown options */
        background-color: var(--backgroundColor, #FFFFFF) !important;
        color: var(--textColor, #333333) !important;
        border: 1px solid var(--secondaryBackgroundColor, #F0F2F6) !important;
    }
    .stSelectbox .st-cd:hover {
        background-color: var(--secondaryBackgroundColor, #F0F2F6) !important;
        color: var(--primaryColor, #4CAF50) !important;
    }

    /* DataFrame (tabella risultati) */
    .stDataFrame {
        background-color: var(--backgroundColor, #FFFFFF) !important;
        border: 1px solid var(--secondaryBackgroundColor, #F0F2F6) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
    }
    .stDataFrame th {
        background-color: var(--secondaryBackgroundColor, #F0F2F6) !important;
        color: var(--textColor, #333333) !important;
        font-weight: bold !important;
    }
    .stDataFrame td {
        color: var(--textColor, #333333) !important;
    }

    /* Spinner di caricamento */
    .stSpinner > div > div {
        color: var(--primaryColor, #4CAF50) !important;
    }
    .stSpinner > div > div > div {
        border-top-color: var(--primaryColor, #4CAF50) !important;
    }

    /* Box di avviso (info, warning, success) */
    .stAlert {
        background-color: var(--secondaryBackgroundColor, #F0F2F6) !important;
        border-left: 5px solid !important; /* Il colore del bordo sarà impostato da Streamlit */
        color: var(--textColor, #333333) !important;
        border-radius: 5px !important;
    }
    /* Colori specifici per i bordi delle alert box */
    .stAlert.success { border-left-color: #4CAF50 !important; }
    .stAlert.warning { border-left-color: #FFC107 !important; }
    .stAlert.info { border-left-color: #2196F3 !important; }

    /* Stili per testo specifico (strong, em) */
    strong {
        color: var(--primaryColor, #4CAF50) !important;
    }
    em {
        color: #9C27B0 !important; /* Viola per enfasi */
    }

    /* Per i grafici Matplotlib, assicurati che lo sfondo sia trasparente per fondersi con il tema Streamlit */
    .stPlotlyChart {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# --- TITOLO E SOTTOTITOLO ---
st.title("📈 Demand Forecasting Dashboard")
st.subheader("Un confronto tra modelli statistici e Machine Learning per la previsione della domanda.")
st.markdown("---")


# ==============================================================================
# 1. FUNZIONI DI GENERAZIONE E PREPARAZIONE DATI
# ==============================================================================

@st.cache_data
def generate_simulated_dataset(num_days=730, num_products=5):
    """Genera un dataset di vendite simulato con stagionalità, trend ed eventi."""
    np.random.seed(42)
    start_date = '2023-01-01'
    date_range = pd.date_range(start=start_date, periods=num_days, freq='D')
    categories = [f'categoria_{chr(65 + i)}' for i in range(num_products)]
    
    data = []
    for date in date_range:
        for category in categories:
            data.append([date, category])
    
    df = pd.DataFrame(data, columns=['data', 'categoria_prodotto'])

    # Logica di simulazione avanzata
    df['quantità_vendute'] = (
        np.random.randint(50, 150, len(df)) +
        20 * np.sin(df['data'].dt.dayofyear * (2 * np.pi / 365)) +
        10 * np.sin(df['data'].dt.dayofweek * (2 * np.pi / 7)) +
        df['data'].dt.year * 0.2 +
        np.random.normal(0, 10, len(df))
    ).astype(int)

    df['quantità_vendute'] = df['quantità_vendute'].apply(lambda x: max(0, x))
    df['è_promo'] = 0
    promo_dates = pd.to_datetime(['2023-03-20', '2023-06-15', '2023-11-25',
                                  '2024-03-20', '2024-06-15', '2024-11-25'])
    df.loc[df['data'].isin(promo_dates), 'è_promo'] = 1
    df.loc[df['data'].isin(promo_dates), 'quantità_vendute'] = (df.loc[df['data'].isin(promo_dates), 'quantità_vendute'] * np.random.uniform(1.5, 2.5)).astype(int)
    
    df['festivo'] = 0
    festive_dates = pd.to_datetime(['2023-12-25', '2024-12-25', '2024-01-01'])
    df.loc[df['data'].isin(festive_dates), 'festivo'] = 1
    
    df['prezzo_unitario'] = np.random.uniform(10.0, 50.0, len(df)).round(2)
    df.loc[df['è_promo'] == 1, 'prezzo_unitario'] = (df.loc[df['è_promo'] == 1, 'prezzo_unitario'] * 0.7).round(2)
    
    df['stockout'] = 0
    df.loc[df['data'].isin(pd.to_datetime(['2024-02-10', '2024-08-20'])), 'stockout'] = 1
    df.loc[df['stockout'] == 1, 'quantità_vendute'] = 0
    
    # Introduce NaN per dimostrare la gestione dei dati mancanti
    df.loc[np.random.randint(0, len(df), 100), 'quantità_vendute'] = np.nan
    
    return df

@st.cache_data
def feature_engineering(df):
    """Applica il feature engineering al dataset."""
    df_copy = df.copy()
    df_copy['data'] = pd.to_datetime(df_copy['data'])
    df_copy = df_copy.sort_values(by=['categoria_prodotto', 'data'])
    
    imputer = SimpleImputer(strategy='mean')
    df_copy['quantità_vendute'] = imputer.fit_transform(df_copy[['quantità_vendute']])

    df_copy['giorno_della_settimana'] = df_copy['data'].dt.weekday
    df_copy['settimana_dell_anno'] = df_copy['data'].dt.isocalendar().week.astype(int)
    df_copy['mese'] = df_copy['data'].dt.month
    df_copy['giorno_dell_anno'] = df_copy['data'].dt.dayofyear
    df_copy['anno'] = df_copy['data'].dt.year

    df_grouped = df_copy.groupby(['categoria_prodotto'])
    
    # Feature laggate e rolling
    df_copy['vendite_lag_1'] = df_grouped['quantità_vendute'].shift(1)
    df_copy['vendite_lag_7'] = df_grouped['quantità_vendute'].shift(7)
    df_copy['media_mobile_7g'] = df_grouped['quantità_vendute'].transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))
    df_copy['std_mobile_7g'] = df_grouped['quantità_vendute'].transform(lambda x: x.rolling(7, min_periods=1).std().shift(1))
    
    # Interazioni e differenze
    df_copy['variazione_prezzo_7g'] = df_grouped['prezzo_unitario'].pct_change(7)
    df_copy['impatto_promo_giorno'] = df_copy['è_promo'] * (df_copy['giorno_della_settimana'] + 1)
    df_copy['interazione_promo_mese'] = df_copy['è_promo'] * df_copy['mese']
    
    df_copy.dropna(inplace=True)
    return df_copy.set_index('data') # Imposta 'data' come indice qui

# ==============================================================================
# 2. FUNZIONI DI ADDESTRAMENTO E VALUTAZIONE
# ==============================================================================

def calculate_metrics(y_true, y_pred):
    """Calcola le metriche di valutazione del modello."""
    # Assicurati che y_pred non abbia valori negativi
    y_pred = np.maximum(0, y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calcolo MAPE, evitando divisione per zero
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape = mape if np.isfinite(mape) else float('inf') # Gestisce casi di divisione per zero o NaN
    
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

def train_and_evaluate_single_model(model_name, train_data, test_data, features):
    """
    Addestra e valuta un singolo modello di previsione.
    Restituisce metriche, previsioni e l'oggetto modello (se ML) per feature importance.
    """
    predictions = np.zeros(len(test_data))
    model_obj = None
    prophet_forecast = None

    y_train = train_data['quantità_vendute']
    y_test = test_data['quantità_vendute']

    if model_name == 'Holt-Winters':
        fit = ExponentialSmoothing(y_train, seasonal_periods=7, trend='add', seasonal='add').fit()
        predictions = fit.forecast(len(y_test))
        model_obj = fit
    
    elif model_name == 'ARIMA':
        try:
            fit = ARIMA(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit()
            predictions = fit.forecast(len(y_test))
            model_obj = fit
        except Exception as e:
            st.warning(f"ARIMA non è stato addestrato per un errore: {e}. Verranno usate previsioni zero.")
            predictions = np.zeros(len(y_test))
            
    elif model_name == 'Prophet':
        df_prophet_train = train_data.reset_index().rename(columns={'data': 'ds', 'quantità_vendute': 'y'})
        prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
        # Aggiungi regressori solo se esistono nel dataframe
        if 'è_promo' in df_prophet_train.columns: prophet_model.add_regressor('è_promo')
        if 'festivo' in df_prophet_train.columns: prophet_model.add_regressor('festivo')
        prophet_model.fit(df_prophet_train)
        
        future = test_data.reset_index().rename(columns={'data': 'ds'})[['ds']]
        if 'è_promo' in test_data.columns: future['è_promo'] = test_data['è_promo'].values
        if 'festivo' in test_data.columns: future['festivo'] = test_data['festivo'].values
        
        prophet_forecast = prophet_model.predict(future)
        predictions = prophet_forecast['yhat'].values
        model_obj = prophet_model # Per Prophet, l'oggetto è il modello stesso
        
    elif model_name == 'LightGBM':
        X_train_ml, X_test_ml = train_data[features], test_data[features]
        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100)
        lgb_model.fit(X_train_ml, y_train)
        predictions = lgb_model.predict(X_test_ml)
        model_obj = lgb_model
        
    elif model_name == 'XGBoost':
        X_train_ml, X_test_ml = train_data[features], test_data[features]
        xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0, n_estimators=100)
        xgb_model.fit(X_train_ml, y_train)
        predictions = xgb_model.predict(X_test_ml)
        model_obj = xgb_model
    
    elif model_name == 'LSTM' and LSTM_AVAILABLE:
        # Preparazione dati per LSTM
        series_lstm_train = y_train.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data_train = scaler.fit_transform(series_lstm_train)

        def create_dataset_lstm(dataset, look_back=7):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back): # Modificato per evitare IndexError
                dataX.append(dataset[i:(i + look_back), 0])
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)
        
        look_back = 7 # Finestra temporale per LSTM
        if len(scaled_data_train) > look_back:
            X_lstm_train, y_lstm_train = create_dataset_lstm(scaled_data_train, look_back)
            X_lstm_train = np.reshape(X_lstm_train, (X_lstm_train.shape[0], 1, X_lstm_train.shape[1]))
            
            model_lstm = Sequential()
            model_lstm.add(LSTM(50, input_shape=(1, look_back)))
            model_lstm.add(Dense(1))
            model_lstm.compile(loss='mean_squared_error', optimizer='adam')
            model_lstm.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=1, verbose=0)
            model_obj = model_lstm

            # Previsione per il test set
            predictions_list = []
            # Inizia la previsione dal punto finale del training data
            current_input = scaled_data_train[-look_back:].reshape(1, 1, look_back)

            for _ in range(len(y_test)):
                pred = model_lstm.predict(current_input, verbose=0)[0]
                predictions_list.append(pred)
                # Aggiorna l'input per la prossima previsione
                current_input = np.append(current_input[:, :, 1:], [[pred]], axis=2)
            
            predictions = scaler.inverse_transform(np.array(predictions_list).reshape(-1, 1)).flatten()
        else:
            st.warning(f"Dati di training insufficienti per il modello LSTM (richiede almeno {look_back + 1} osservazioni).")
            predictions = np.zeros(len(y_test))
            model_obj = None
    else:
        st.warning(f"Modello {model_name} non supportato o LSTM non disponibile.")
        predictions = np.zeros(len(y_test))
    
    metrics = calculate_metrics(y_test, predictions)
    return metrics, predictions, model_obj, prophet_forecast

# ==============================================================================
# 3. FUNZIONI DI VISUALIZZAZIONE
# ==============================================================================

def plot_performance_comparison(results_df):
    """Genera un grafico comparativo delle metriche dei modelli."""
    fig, ax = plt.subplots(figsize=(12, 7))
    # Usiamo lo stile Streamlit per Matplotlib per coerenza con il tema
    # plt.style.use('ggplot') # Non usiamo più stili globali qui
    
    # Prepara i dati per il plot
    metrics_to_plot = ['RMSE', 'MAE', 'MAPE', 'R2']
    plot_df = results_df[metrics_to_plot].copy()
    
    # Normalizza le metriche per visualizzarle sulla stessa scala (opzionale, ma utile per confronto)
    # Per R2, vogliamo massimizzare, per gli altri minimizzare.
    # Quindi invertiamo RMSE, MAE, MAPE per la normalizzazione visiva se necessario.
    # Per semplicità, plottiamo i valori reali e lasciamo all'utente l'interpretazione.

    plot_df.plot(kind='bar', ax=ax, colormap='viridis', alpha=0.8) # Usiamo una colormap
    
    ax.set_title('Confronto delle Performance dei Modelli', fontsize=16, color=st.get_option("theme.textColor"))
    ax.set_xlabel('Modelli', fontsize=12, color=st.get_option("theme.textColor"))
    ax.set_ylabel('Valore della Metrica', fontsize=12, color=st.get_option("theme.textColor"))
    ax.tick_params(axis='x', rotation=45, colors=st.get_option("theme.textColor"))
    ax.tick_params(axis='y', colors=st.get_option("theme.textColor"))
    ax.set_facecolor(st.get_option("theme.secondaryBackgroundColor")) # Sfondo del plot
    fig.patch.set_facecolor(st.get_option("theme.backgroundColor")) # Sfondo della figura

    # Aggiusta i bordi degli assi
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(st.get_option("theme.textColor"))
    ax.spines['bottom'].set_color(st.get_option("theme.textColor"))

    plt.tight_layout()
    st.pyplot(fig)


def plot_predictions_vs_actual(df_test, predictions_dict, selected_product_name):
    """Genera un grafico delle previsioni vs. valori reali per ogni modello."""
    fig = go.Figure()
    
    # Valori Reali
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['quantità_vendute'], mode='lines', 
                             name='Valori Reali', line=dict(color='#4CAF50', width=3))) # Verde primario
    
    # Colori per le previsioni (possono essere personalizzati)
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF'] # Esempio di colori vivaci
    
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        fig.add_trace(go.Scatter(x=df_test.index, y=predictions, mode='lines', 
                                 name=f'Previsioni {model_name}', line=dict(color=colors[i], dash='dash')))
            
    fig.update_layout(
        title_text=f'Previsioni vs. Valori Reali per {selected_product_name}',
        xaxis_title='Data',
        yaxis_title='Quantità Vendute',
        hovermode="x unified",
        # Aggiorna il tema per riflettere il tema chiaro di Streamlit
        template="plotly_white", # Usa un tema chiaro di Plotly
        font=dict(color=st.get_option("theme.textColor")),
        plot_bgcolor=st.get_option("theme.backgroundColor"), # Sfondo del plot
        paper_bgcolor=st.get_option("theme.backgroundColor"), # Sfondo della carta
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)", # Sfondo leggermente trasparente per la legenda
            bordercolor="#E0E0E0",
            borderwidth=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(model_obj, features):
    """Genera un grafico dell'importanza delle feature per un modello ML."""
    if isinstance(model_obj, (lgb.LGBMRegressor, xgb.XGBRegressor)):
        importance = model_obj.feature_importances_
        model_name = "LightGBM" if isinstance(model_obj, lgb.LGBMRegressor) else "XGBoost"
        
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette='viridis')
        
        ax.set_title(f"Importanza delle Feature per il modello {model_name}", fontsize=16, color=st.get_option("theme.textColor"))
        ax.set_xlabel("Importanza", fontsize=12, color=st.get_option("theme.textColor"))
        ax.set_ylabel("Feature", fontsize=12, color=st.get_option("theme.textColor"))
        ax.tick_params(axis='x', colors=st.get_option("theme.textColor"))
        ax.tick_params(axis='y', colors=st.get_option("theme.textColor"))
        ax.set_facecolor(st.get_option("theme.secondaryBackgroundColor")) # Sfondo del plot
        fig.patch.set_facecolor(st.get_option("theme.backgroundColor")) # Sfondo della figura

        # Aggiusta i bordi degli assi
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(st.get_option("theme.textColor"))
        ax.spines['bottom'].set_color(st.get_option("theme.textColor"))

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Importanza delle feature non disponibile per questo tipo di modello.")

def plot_seasonal_decomposition(df_train_single, selected_product_name):
    """Genera i grafici di decomposizione stagionale."""
    st.subheader(f"Analisi della Stagionalità per {selected_product_name}")
    st.markdown("La decomposizione della serie temporale separa i dati in tre componenti: **Trend**, **Stagionalità** e **Residuo**, offrendo una comprensione più approfondita dei pattern di vendita ricorrenti.")

    try:
        if len(df_train_single) < 2 * 7: # Richiede almeno due cicli completi per period=7
            st.warning("Dati di training insufficienti per una decomposizione stagionale significativa (richiede almeno 14 giorni).")
        else:
            result = seasonal_decompose(df_train_single['quantità_vendute'], model='additive', period=7)
            
            fig, ax = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
            fig.patch.set_facecolor(st.get_option("theme.backgroundColor")) # Sfondo della figura

            # Plot styling
            styles = [
                {'color': '#4CAF50', 'label': 'Originale'}, # Verde primario
                {'color': '#FF9800', 'label': 'Trend'},    # Arancione
                {'color': '#2196F3', 'label': 'Stagionalità'}, # Blu
                {'color': '#9C27B0', 'label': 'Residuo'}   # Viola
            ]
            
            for i, (plot_data, style) in enumerate(zip([result.observed, result.trend, result.seasonal, result.resid], styles)):
                plot_data.plot(ax=ax[i], legend=False, color=style['color'])
                ax[i].set_ylabel(style['label'], fontsize=12, color=st.get_option("theme.textColor"))
                ax[i].tick_params(axis='x', colors=st.get_option("theme.textColor"))
                ax[i].tick_params(axis='y', colors=st.get_option("theme.textColor"))
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
                ax[i].spines['left'].set_color(st.get_option("theme.textColor"))
                ax[i].spines['bottom'].set_color(st.get_option("theme.textColor"))
                ax[i].set_facecolor(st.get_option("theme.secondaryBackgroundColor")) # Sfondo del grafico interno
                ax[i].set_xlabel("Data", color=st.get_option("theme.textColor"))
                
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Errore durante la decomposizione della serie storica: `{e}`. Controlla che i dati siano sufficienti e non contengano valori non numerici.")


# ==============================================================================
# 4. FUNZIONE PRINCIPALE (MAIN) E LOGICA STREAMLIT
# ==============================================================================

# Inizializzazione dello stato della sessione
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'last_run_params' not in st.session_state:
    st.session_state.last_run_params = {}

# Recupera i parametri dalla sidebar (con chiavi per evitare warning)
current_product_name = st.session_state.selected_product_sidebar
current_test_days = st.session_state.test_days_sidebar
current_unit_margin = st.session_state.unit_margin_sidebar
current_overstock_daily_cost = st.session_state.overstock_cost_sidebar

# Controlla se i parametri che influenzano l'addestramento sono cambiati
params_changed = (
    st.session_state.last_run_params.get('product_name') != current_product_name or
    st.session_state.last_run_params.get('test_days') != current_test_days
)

# Se il bottone "Avvia Analisi" è stato cliccato O i parametri chiave sono cambiati,
# e non ci sono risultati cached, o i parametri chiave sono cambiati
if st.session_state.run_analysis_button or (params_changed and st.session_state.analysis_results is not None):
    st.session_state.run_analysis_button = False # Reset del bottone
    
    # Aggiorna i parametri dell'ultima esecuzione
    st.session_state.last_run_params = {
        'product_name': current_product_name,
        'test_days': current_test_days
    }

    with st.spinner("Avvio dell'analisi... potrebbe volerci qualche secondo."):
        df = generate_simulated_dataset()
        df_single = df[df['categoria_prodotto'] == current_product_name].copy() # Usa categoria_prodotto

        # Split training/testing
        if df_single.empty:
            st.error("Il dataset per il prodotto selezionato è vuoto. Controlla la selezione del prodotto o il dataset.")
            st.session_state.analysis_results = None
        else:
            train_end_date = df_single.index.max() - pd.Timedelta(days=current_test_days)
            
            df_train_single = df_single.loc[df_single.index <= train_end_date].copy()
            df_test_single = df_single.loc[df_single.index > train_end_date].copy()

            if df_train_single.empty or df_test_single.empty:
                st.error("I dataset di training o di test sono vuoti dopo lo split. Prova a modificare il 'Numero di giorni da prevedere' o controlla la lunghezza totale del dataset simulato.")
                st.session_state.analysis_results = None
            else:
                df_with_features = feature_engineering(df_single) # Applica feature engineering
                
                df_train_with_features = df_with_features.loc[df_with_features.index <= train_end_date].copy()
                df_test_with_features = df_with_features.loc[df_with_features.index > train_end_date].copy()

                if df_train_with_features.empty or df_test_with_features.empty:
                    st.error("I dataset di training o di test con features sono vuoti. Questo può accadere se ci sono troppi NaN all'inizio della serie dopo la creazione delle features. Prova a usare un periodo di training più lungo o un dataset più grande.")
                    st.session_state.analysis_results = None
                else:
                    features = [
                        'giorno_della_settimana', 'settimana_dell_anno', 'mese', 'giorno_dell_anno', 'anno',
                        'vendite_lag_1', 'vendite_lag_7', 'media_mobile_7g', 'std_mobile_7g',
                        'variazione_prezzo_7g', 'è_promo', 'festivo', 'stockout', 'impatto_promo_giorno', 'interazione_promo_mese'
                    ]
                    # Filtra le features_to_use per assicurarci che esistano nel dataframe
                    available_features = [f for f in features if f in df_train_with_features.columns]

                    models_to_run = ["Holt-Winters", "ARIMA", "Prophet", "LightGBM", "XGBoost"]
                    if LSTM_AVAILABLE:
                        models_to_run.append("LSTM")

                    results_metrics = []
                    predictions_dict = {}
                    model_objects = {} # Per salvare gli oggetti modello per feature importance
                    prophet_forecast_data = {} # Per salvare i dati di forecast di Prophet

                    for model_name in models_to_run:
                        if model_name in ["LightGBM", "XGBoost", "LSTM"]:
                            metrics, predictions, model_obj, prophet_fc = train_and_evaluate_single_model(
                                model_name, df_train_with_features, df_test_with_features, available_features
                            )
                        else: # Holt-Winters, ARIMA, Prophet
                            metrics, predictions, model_obj, prophet_fc = train_and_evaluate_single_model(
                                model_name, df_train_single, df_test_single, []
                            )
                        
                        results_metrics.append({
                            "Modello": model_name,
                            "RMSE": metrics['RMSE'],
                            "MAE": metrics['MAE'],
                            "MAPE": metrics['MAPE'],
                            "R2": metrics['R2']
                        })
                        predictions_dict[model_name] = predictions
                        model_objects[model_name] = model_obj
                        if prophet_fc is not None:
                            prophet_forecast_data[model_name] = prophet_fc

                    results_df_metrics = pd.DataFrame(results_metrics).set_index("Modello")
                    
                    st.session_state.analysis_results = {
                        "results_df_metrics": results_df_metrics,
                        "predictions_dict": predictions_dict,
                        "model_objects": model_objects,
                        "df_test_single": df_test_single,
                        "df_train_single": df_train_single,
                        "selected_product_name": current_product_name,
                        "features_for_ml": available_features, # Salva le feature usate per ML
                        "prophet_forecast_data": prophet_forecast_data
                    }
                    st.success("Analisi completata!")
    
# Visualizzazione dei risultati se disponibili
if st.session_state.analysis_results is not None:
    results_df_metrics = st.session_state.analysis_results["results_df_metrics"]
    predictions_dict = st.session_state.analysis_results["predictions_dict"]
    model_objects = st.session_state.analysis_results["model_objects"]
    df_test_single = st.session_state.analysis_results["df_test_single"]
    df_train_single = st.session_state.analysis_results["df_train_single"]
    selected_product_name_display = st.session_state.analysis_results["selected_product_name"]
    features_for_ml = st.session_state.analysis_results["features_for_ml"]
    prophet_forecast_data = st.session_state.analysis_results["prophet_forecast_data"]

    # Ricalcola i costi economici con i parametri correnti (se cambiati dopo il run)
    economic_results = []
    for model_name, predictions in predictions_dict.items():
        total_cost, understock_cost, overstock_cost = calculate_business_impact(
            predictions, df_test_single['quantità_vendute'].values, current_unit_margin, current_overstock_daily_cost
        )
        economic_results.append({
            "Modello": model_name,
            "Costo Totale (€)": total_cost,
            "Costo Understock (€)": understock_cost,
            "Costo Overstock (€)": overstock_cost
        })
    economic_df = pd.DataFrame(economic_results).set_index("Modello")

    st.header("🔍 Confronto Risultati")
    st.markdown("Questa tabella riassume le performance di ogni modello in termini di metriche statistiche e costo economico.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metriche di Previsione")
        st.dataframe(results_df_metrics.style.background_gradient(cmap='Greens', subset=['R2'], axis=0).background_gradient(cmap='Reds_r', subset=['RMSE', 'MAE', 'MAPE'], axis=0))
    with col2:
        st.subheader("Impatto Economico")
        st.dataframe(economic_df.style.background_gradient(cmap='Reds_r', subset=['Costo Totale (€)'], axis=0))

    st.markdown("---")

    st.header("📊 Previsioni vs. Valori Reali")
    st.markdown("Questo grafico visualizza il confronto tra le vendite reali e le previsioni di ogni modello nel periodo di test.")
    plot_predictions_vs_actual(df_test_single, predictions_dict, selected_product_name_display)

    st.markdown("---")

    st.header("📈 Analisi della Stagionalità")
    plot_seasonal_decomposition(df_train_single, selected_product_name_display)

    st.markdown("---")

    st.header("💡 Importanza delle Feature")
    st.markdown("Questo grafico mostra quali variabili di input hanno avuto maggiore peso nella previsione per i modelli di Machine Learning selezionati.")
    
    # Seleziona il modello per l'importanza delle feature
    ml_models_for_feature_importance = [m for m in ["LightGBM", "XGBoost"] if m in model_objects and model_objects[m] is not None]
    if ml_models_for_feature_importance:
        selected_ml_model = st.selectbox(
            "Seleziona il modello ML per visualizzare l'importanza delle feature:",
            ml_models_for_feature_importance,
            key="feature_importance_selector"
        )
        plot_feature_importance(model_objects[selected_ml_model], features_for_ml)
    else:
        st.info("Nessun modello ML disponibile per visualizzare l'importanza delle feature.")

else:
    st.info("Clicca 'Avvia Analisi' nella sidebar per iniziare l'elaborazione dei dati e la previsione.")
