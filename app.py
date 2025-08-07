import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import datetime

# --- CONFIGURAZIONE PAGINA STREAMLIT E STILI CSS PERSONALIZZATI ---
st.set_page_config(layout="wide", page_title="Demand Forecasting Dashboard", page_icon="📈")

st.markdown("""
<head>
    <!-- JetBrains Mono font for that terminal feel -->
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,700;1,400&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<style>
    /* Definizione delle variabili CSS per il tema terminale */
    :root {
        --terminal-green: #00ff41;
        --terminal-blue: #0066ff;
        --terminal-orange: #ff6600;
        --terminal-purple: #cc66ff;
        --terminal-bg: #0a0a0a;
        --terminal-secondary: #1a1a1a;
        --terminal-accent: #2a2a2a;
    }
    
    /* Stili globali per il body e il container principale di Streamlit */
    html, body {
        height: 100%;
        margin: 0;
        padding: 0;
    }

    .stApp {
        background: linear-gradient(135deg, var(--terminal-bg) 0%, #111 50%, var(--terminal-secondary) 100%) !important;
        background-attachment: fixed !important;
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.7 !important;
        padding-top: 2rem !important;
        padding-right: 2rem !important;
        padding-left: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Intestazioni */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--terminal-green) !important;
        text-shadow: 0 0 5px rgba(0, 255, 65, 0.3) !important;
    }

    /* Testo generale */
    p, .stText, .stMarkdown, label {
        color: #e0e0e0 !important;
    }

    /* Sidebar */
    .stSidebar {
        background: var(--terminal-secondary) !important;
        border-right: 1px solid var(--terminal-accent) !important;
        color: #e0e0e0 !important;
    }
    .stSidebar .stSelectbox label, .stSidebar .stSlider label, .stSidebar h2 {
        color: var(--terminal-blue) !important;
    }

    /* Bottoni */
    .stButton>button {
        background-color: var(--terminal-blue) !important;
        color: white !important;
        border: 1px solid var(--terminal-green) !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: bold !important;
        box-shadow: 0 0 10px rgba(0, 102, 255, 0.3) !important;
    }
    .stButton>button:hover {
        background-color: var(--terminal-purple) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 0 15px rgba(204, 102, 255, 0.5) !important;
    }

    /* Slider */
    .stSlider .st-bd { /* Track */
        background: var(--terminal-accent) !important;
    }
    .stSlider .st-be { /* Fill */
        background: var(--terminal-green) !important;
    }
    .stSlider .st-bf { /* Thumb */
        background: var(--terminal-blue) !important;
        border: 2px solid var(--terminal-green) !important;
    }

    /* Selectbox */
    .stSelectbox .st-bb { /* Dropdown arrow */
        color: var(--terminal-green) !important;
    }
    .stSelectbox .st-cc { /* Selected value text */
        color: #e0e0e0 !important;
        background-color: var(--terminal-accent) !important;
        border: 1px solid var(--terminal-green) !important;
    }
    .stSelectbox .st-cd { /* Dropdown options */
        background-color: var(--terminal-secondary) !important;
        color: #e0e0e0 !important;
        border: 1px solid var(--terminal-green) !important;
    }
    .stSelectbox .st-cd:hover {
        background-color: var(--terminal-accent) !important;
        color: var(--terminal-green) !important;
    }

    /* DataFrame (tabella risultati) */
    .stDataFrame {
        background-color: var(--terminal-bg) !important;
        color: #e0e0e0 !important;
        border: 1px solid var(--terminal-accent) !important;
        border-radius: 8px !important;
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.3) !important;
    }
    .stDataFrame th {
        background-color: var(--terminal-accent) !important;
        color: var(--terminal-green) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stDataFrame td {
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Spinner di caricamento */
    .stSpinner > div > div {
        color: var(--terminal-green) !important;
    }
    .stSpinner > div > div > div {
        border-top-color: var(--terminal-green) !important;
    }

    /* Box di avviso (info, warning, success) */
    .stAlert {
        background-color: var(--terminal-accent) !important;
        border-left: 5px solid var(--terminal-blue) !important;
        color: #e0e0e0 !important;
    }
    .stAlert.success { border-left-color: var(--terminal-green) !important; }
    .stAlert.warning { border-left-color: var(--terminal-orange) !important; }
    .stAlert.info { border-left-color: var(--terminal-blue) !important; }

    /* Stili per testo specifico */
    strong {
        color: var(--terminal-orange) !important;
    }
    em {
        color: var(--terminal-purple) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- TITOLO E SOTTOTITOLO ---
st.title("📈 Demand Forecasting Dashboard")
st.subheader("Un confronto tra modelli statistici e Machine Learning per la previsione della domanda.")
st.markdown("---")


# --- SEZIONE GENERAZIONE DATI (con caching) ---
@st.cache_data
def generate_simulated_dataset(num_days=730, num_products=5):
    """
    Genera dati che simulano la realtà delle vendite con trend, stagionalità e rumore.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=num_days, freq="D")
    data = []
    product_names = [f"Prodotto_{i}" for i in range(1, num_products + 1)]
    for product_id, product_name in enumerate(product_names):
        # Componenti di base
        trend = np.linspace(0, 150, num_days) + np.random.normal(0, 10, num_days)
        weekly_seasonality = np.sin(np.arange(num_days) / 7 * 2 * np.pi) * 20
        yearly_seasonality = np.sin(np.arange(num_days) / 365.25 * 2 * np.pi) * 50
        
        # Eventi speciali (simulati)
        promotions = np.zeros(num_days)
        promo_days = [30, 90, 180, 270, 360, 450, 540, 630, 720] # Esempio di giorni con promozioni
        for day_idx in promo_days:
            if day_idx < num_days:
                promotions[day_idx] = 1
        promotions_effect = promotions * 100
        
        # Rumore realistico
        noise = np.random.normal(0, 15, num_days)
        
        # Quantità totali
        quantities = np.maximum(0, trend + weekly_seasonality + yearly_seasonality + promotions_effect + noise)
        
        product_data = pd.DataFrame({
            "data": dates,
            "quantità_vendute": quantities.round().astype(int),
            "id_prodotto": product_id,
            "nome_prodotto": product_name
        })
        data.append(product_data)
    
    df = pd.concat(data, ignore_index=True)
    df['data'] = pd.to_datetime(df['data'])
    df = df.set_index('data')
    return df

# --- SEZIONE FEATURE ENGINEERING ---
def create_features(df):
    """
    Crea nuove feature dalla data per catturare pattern temporali.
    Aggiunge anche un'imputazione semplice per le quantità vendute.
    """
    df_copy = df.copy()
    
    # Imputazione dei valori mancanti (se presenti, anche se il simulatore non li crea)
    if df_copy['quantità_vendute'].isnull().any():
        df_copy['quantità_vendute'] = df_copy['quantità_vendute'].fillna(df_copy['quantità_vendute'].mean())

    df_copy['giorno_della_settimana'] = df_copy.index.weekday
    df_copy['settimana_dell_anno'] = df_copy.index.isocalendar().week.astype(int)
    df_copy['mese'] = df_copy.index.month
    df_copy['giorno_dell_anno'] = df_copy.index.dayofyear
    df_copy['anno'] = df_copy.index.year
    df_copy['is_weekend'] = df_copy['giorno_della_settimana'].isin([5, 6]).astype(int)
    
    # Lag features (necessitano di essere calcolate per ogni prodotto se il DF contiene più prodotti)
    # Assicurati che il dataframe sia già filtrato per singolo prodotto o raggruppa
    # In questo contesto, df_copy è già per singolo prodotto
    df_copy['vendite_lag_1'] = df_copy['quantità_vendute'].shift(1)
    df_copy['vendite_lag_7'] = df_copy['quantità_vendute'].shift(7)
    
    # Rolling statistics
    df_copy['media_mobile_7g'] = df_copy['quantità_vendute'].rolling(window=7, min_periods=1).mean().shift(1)
    
    # Rimuovi eventuali NaN introdotti dalle lag/rolling features all'inizio della serie
    return df_copy.dropna()

# --- SEZIONE MODELLAZIONE E VALUTAZIONE ---
def train_and_evaluate_model(model_name, train_data, test_data, features):
    """
    Addestra e valuta un modello di previsione.
    """
    predictions = np.zeros(len(test_data))
    prophet_forecast = None
    
    # Prepara i dati
    y_train = train_data['quantità_vendute']
    y_test = test_data['quantità_vendute']
    
    if model_name == 'Holt-Winters':
        # Holt-Winters non usa features esplicite, solo la serie temporale
        fit = ExponentialSmoothing(y_train, seasonal_periods=7, trend='add', seasonal='add').fit()
        predictions = fit.forecast(len(y_test))
    
    elif model_name == 'ARIMA':
        try:
            # ARIMA non usa features esplicite, solo la serie temporale
            fit = ARIMA(y_train, order=(1,1,1), seasonal_order=(1,1,1,7)).fit()
            predictions = fit.forecast(len(y_test))
        except Exception as e:
            st.warning(f"ARIMA non è stato addestrato per un errore: {e}. Verranno usate previsioni zero.")
            predictions = np.zeros(len(y_test))
            
    elif model_name == 'Prophet':
        # Prophet richiede un DataFrame con colonne 'ds' (data) e 'y' (valore)
        df_prophet_train = train_data.reset_index().rename(columns={'data': 'ds', 'quantità_vendute': 'y'})
        prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
        prophet_model.fit(df_prophet_train)
        
        # Prepara il DataFrame futuro per Prophet
        future = test_data.reset_index().rename(columns={'data': 'ds'})[['ds']]
        prophet_forecast = prophet_model.predict(future)
        predictions = prophet_forecast['yhat'].values
        
    else: # LightGBM e XGBoost
        # Questi modelli usano le features esplicite
        X_train = train_data[features]
        X_test = test_data[features]
        
        if model_name == 'LightGBM':
            lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100)
            lgb_model.fit(X_train, y_train)
            predictions = lgb_model.predict(X_test)
        
        elif model_name == 'XGBoost':
            xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0, n_estimators=100)
            xgb_model.fit(X_train, y_train)
            predictions = xgb_model.predict(X_test)
            
    # Assicurati che le previsioni non siano negative
    predictions = np.maximum(0, predictions)
    
    # Calcolo delle metriche
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'MAE': mean_absolute_error(y_test, predictions),
        'R2': r2_score(y_test, predictions),
        # Gestione divisione per zero per MAPE
        'MAPE': np.mean(np.abs((y_test - predictions) / y_test)) * 100 if np.all(y_test != 0) else float('inf')
    }
    
    return metrics, predictions, prophet_forecast

# --- SEZIONE CALCOLO IMPATTO ECONOMICO ---
def calculate_business_impact(predictions, actuals, unit_margin, overstock_daily_cost):
    """
    Calcola l'impatto economico degli errori di previsione.
    """
    predictions = np.maximum(0, predictions)
    
    # Understock: profitto perso
    understock_units = np.maximum(0, actuals - predictions)
    understock_cost = np.sum(understock_units) * unit_margin
    
    # Overstock: costi di mantenimento
    overstock_units = np.maximum(0, predictions - actuals)
    overstock_cost = np.sum(overstock_units) * overstock_daily_cost
    
    total_cost = understock_cost + overstock_cost
    return total_cost, understock_cost, overstock_cost

# --- INTERFACCIA UTENTE STREAMLIT ---
with st.sidebar:
    st.header("⚙️ Parametri del Modello")
    
    selected_product_name = st.selectbox(
        "Seleziona il Prodotto da Analizzare",
        [f"Prodotto_{i}" for i in range(1, 6)]
    )
    
    # Imposta un valore minimo per test_days per garantire dati sufficienti per il training
    # Ad esempio, se hai 730 giorni di dati, e vuoi almeno 365 giorni per il training,
    # il massimo test_days sarà 730 - 365 = 365.
    # Il dataset simulato ha 730 giorni.
    # Assicurati che ci siano almeno 100 giorni per il training.
    max_test_days = 730 - 100 
    test_days = st.slider("Numero di giorni da prevedere (Test Set)", 30, max_test_days, 30)
    
    st.markdown("---")
    st.header("📊 Parametri Economici")
    unit_margin = st.number_input("Margine per Unità Venduta (€)", value=10.0, step=0.5)
    overstock_daily_cost = st.number_input("Costo Mantenimento Magazzino (€/unità/giorno)", value=0.5, step=0.1)

    # Bottone di avvio
    st.markdown("---")
    if st.button("Avvia Analisi"):
        st.session_state.run_analysis = True
    
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False


if st.session_state.run_analysis:
    st.info("Avvio dell'analisi... potrebbe volerci qualche secondo.")
    
    # Caricamento e preparazione dei dati
    df = generate_simulated_dataset()
    df_single = df[df['nome_prodotto'] == selected_product_name].copy()
    
    # Split training/testing
    if df_single.empty:
        st.error("Il dataset per il prodotto selezionato è vuoto. Controlla la selezione del prodotto o il dataset.")
        st.session_state.run_analysis = False
    else:
        train_end_date = df_single.index.max() - pd.Timedelta(days=test_days)
        
        df_train_single = df_single.loc[df_single.index <= train_end_date].copy()
        df_test_single = df_single.loc[df_single.index > train_end_date].copy()

        if df_train_single.empty or df_test_single.empty:
            st.error("I dataset di training o di test sono vuoti dopo lo split. Prova a modificare il 'Numero di giorni da prevedere' o controlla la lunghezza totale del dataset simulato.")
            st.session_state.run_analysis = False
        else:
            # Feature Engineering
            df_with_features = create_features(df_single)
            
            # Ora risplitta il dataframe con le features
            df_train_with_features = df_with_features.loc[df_with_features.index <= train_end_date].copy()
            df_test_with_features = df_with_features.loc[df_with_features.index > train_end_date].copy()

            # Assicurati che i dataframe con features non siano vuoti dopo il dropna
            if df_train_with_features.empty or df_test_with_features.empty:
                st.error("I dataset di training o di test con features sono vuoti. Questo può accadere se ci sono troppi NaN all'inizio della serie dopo la creazione delle features. Prova a usare un periodo di training più lungo o un dataset più grande.")
                st.session_state.run_analysis = False
            else:
                features_to_use = [
                    'giorno_della_settimana', 'settimana_dell_anno', 'mese', 'is_weekend', 
                    'vendite_lag_1', 'vendite_lag_7', 'media_mobile_7g'
                ]
                # Filtra le features_to_use per assicurarci che esistano nel dataframe
                available_features = [f for f in features_to_use if f in df_train_with_features.columns]


                models = ["Holt-Winters", "ARIMA", "Prophet", "LightGBM", "XGBoost"]
                results = []
                predictions_dict = {}
                prophet_forecast_dict = {}

                # Esecuzione dei modelli
                with st.spinner("Addestramento e valutazione dei modelli..."):
                    for model_name in models:
                        # Passa il dataframe corretto (con o senza features) a seconda del modello
                        if model_name in ["LightGBM", "XGBoost"]:
                            metrics, predictions, prophet_forecast = train_and_evaluate_model(
                                model_name, df_train_with_features, df_test_with_features, available_features
                            )
                        else: # Holt-Winters, ARIMA, Prophet
                            metrics, predictions, prophet_forecast = train_and_evaluate_model(
                                model_name, df_train_single, df_test_single, [] # Non usano features esplicite
                            )

                        total_cost, _, _ = calculate_business_impact(
                            predictions, df_test_single['quantità_vendute'].values, unit_margin, overstock_daily_cost
                        )
                        
                        results.append({
                            "Modello": model_name,
                            "RMSE": metrics['RMSE'],
                            "MAE": metrics['MAE'],
                            "R2": metrics['R2'],
                            "Costo Totale (€)": total_cost
                        })
                        predictions_dict[model_name] = predictions
                        if prophet_forecast is not None:
                            prophet_forecast_dict[model_name] = prophet_forecast

                results_df = pd.DataFrame(results).set_index("Modello")
                
                st.success("Analisi completata!")

                # --- SEZIONE RISULTATI E GRAFICI ---
                
                st.header("🔍 Confronto Risultati")
                st.markdown("Questa tabella riassume le performance di ogni modello in termini di metriche statistiche e costo economico.")
                st.dataframe(results_df.style.background_gradient(cmap='viridis', subset=['RMSE', 'MAE', 'Costo Totale (€)'], axis=0).highlight_max(axis=0, subset=['R2'], color='green'))

                st.markdown("---")

                st.header("📊 Previsioni vs. Valori Reali")
                st.markdown("Questo grafico visualizza il confronto tra le vendite reali e le previsioni di ogni modello nel periodo di test.")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_test_single.index, y=df_test_single['quantità_vendute'], mode='lines', name='Valori Reali', line=dict(color='#0066ff', width=3)))
                
                colors = ['#ff6b9d', '#c3e88d', '#fab387', '#89dceb', '#cc66ff']
                
                for i, (model_name, predictions) in enumerate(predictions_dict.items()):
                    fig.add_trace(go.Scatter(x=df_test_single.index, y=predictions, mode='lines', name=f'Previsioni {model_name}', line=dict(color=colors[i], dash='dash')))

                if 'Prophet' in prophet_forecast_dict:
                    prophet_forecast = prophet_forecast_dict['Prophet']
                    fig.add_trace(go.Scatter(
                        x=df_test_single.index,
                        y=prophet_forecast['yhat_lower'].values,
                        line=dict(width=0),
                        mode='lines',
                        marker=dict(color="#444"),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_test_single.index,
                        y=prophet_forecast['yhat_upper'].values,
                        fill='tonexty',
                        fillcolor='rgba(108, 112, 134, 0.2)',
                        line=dict(width=0),
                        mode='lines',
                        marker=dict(color="#444"),
                        name='Intervallo di Confidenza Prophet'
                    ))

                fig.update_layout(
                    title_text=f"Previsioni vs. Valori Reali per {selected_product_name}",
                    template="plotly_dark",
                    xaxis_title="Data",
                    yaxis_title="Quantità Vendute",
                    font=dict(color='#e0e0e0'),
                    legend=dict(x=1.02, y=1, bgcolor='rgba(42, 42, 42, 0.5)', bordercolor='#00ff41', borderwidth=1),
                    plot_bgcolor='rgba(10, 10, 10, 0.5)',
                    paper_bgcolor='rgba(10, 10, 10, 0.5)',
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                st.header("📊 Decomposizione della Serie Storica")
                st.markdown("Questa analisi scompone la serie storica del training set in Trend, Stagionalità e Residuo.")
                
                try:
                    # Ho provato a usare period=7 per la decomposizione, che è più robusto con dataset più corti
                    # Se il dataset di training è molto corto, anche period=7 potrebbe fallire.
                    # In un'app reale, si dovrebbe controllare la lunghezza del train_data prima di chiamare seasonal_decompose.
                    if len(df_train_single) < 2 * 7: # Richiede almeno due cicli completi per period=7
                        st.warning("Dati di training insufficienti per una decomposizione stagionale significativa (richiede almeno 14 giorni).")
                    else:
                        result = seasonal_decompose(df_train_single['quantità_vendute'], model='additive', period=7)
                        
                        fig, ax = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
                        fig.patch.set_facecolor('#0a0a0a')
                        
                        # Plot styling
                        styles = [
                            {'color': '#0066ff', 'label': 'Originale'},
                            {'color': '#ff6600', 'label': 'Trend'},
                            {'color': '#cc66ff', 'label': 'Stagionalità'},
                            {'color': '#00ff41', 'label': 'Residuo'}
                        ]
                        
                        for i, (plot_data, style) in enumerate(zip([result.observed, result.trend, result.seasonal, result.resid], styles)):
                            plot_data.plot(ax=ax[i], legend=False, color=style['color'])
                            ax[i].set_ylabel(style['label'], fontsize=12, color=style['color'])
                            ax[i].tick_params(axis='x', colors='#e0e0e0')
                            ax[i].tick_params(axis='y', colors='#e0e0e0')
                            ax[i].spines['top'].set_visible(False)
                            ax[i].spines['right'].set_visible(False)
                            ax[i].spines['left'].set_color('#3a3a3a')
                            ax[i].spines['bottom'].set_color('#3a3a3a')
                            ax[i].set_facecolor('#1a1a1a')
                            ax[i].set_xlabel("Data", color='#e0e0e0')
                            
                        fig.tight_layout()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Errore durante la decomposizione della serie storica: `{e}`. Controlla che i dati siano sufficienti e non contengano valori non numerici.")
else:
    st.info("Clicca 'Avvia Analisi' nella sidebar per iniziare.")
