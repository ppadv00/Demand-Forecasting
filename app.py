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

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
st.set_page_config(layout="wide", page_title="Demand Forecasting Dashboard", page_icon="📈")
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #00ff41;
    }
    p, .stText {
        color: #e0e0e0;
    }
    .stSelectbox label, .stSlider label {
        color: #89dceb;
    }
    .stButton>button {
        background-color: #0066ff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #0052cc;
        transform: translateY(-2px);
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
        christmas_dates = [d for d in dates if d.month == 12 and d.day > 15]
        for d in christmas_dates:
            promotions[dates.get_loc(d)] = 1
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
    """
    df_copy = df.copy()
    df_copy['giorno_della_settimana'] = df_copy.index.weekday
    df_copy['settimana_dell_anno'] = df_copy.index.isocalendar().week.astype(int)
    df_copy['mese'] = df_copy.index.month
    df_copy['giorno_dell_anno'] = df_copy.index.dayofyear
    df_copy['anno'] = df_copy.index.year
    df_copy['is_weekend'] = df_copy['giorno_della_settimana'].isin([5, 6]).astype(int)
    
    # Lag features
    df_copy['vendite_lag_1'] = df_copy['quantità_vendute'].shift(1)
    df_copy['vendite_lag_7'] = df_copy['quantità_vendute'].shift(7)
    
    # Rolling statistics
    df_copy['media_mobile_7g'] = df_copy['quantità_vendute'].rolling(window=7).mean().shift(1)
    
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
        fit = ExponentialSmoothing(y_train, seasonal_periods=7, trend='add', seasonal='add').fit()
        predictions = fit.forecast(len(test_data))
    
    elif model_name == 'ARIMA':
        try:
            fit = ARIMA(y_train, order=(1,1,1), seasonal_order=(1,1,1,7)).fit()
            predictions = fit.forecast(len(test_data))
        except Exception:
            st.warning(f"ARIMA non è stato addestrato per un errore.")
            predictions = np.zeros(len(test_data))
            
    elif model_name == 'Prophet':
        df_prophet_train = train_data.reset_index().rename(columns={'data': 'ds', 'quantità_vendute': 'y'})
        prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
        prophet_model.fit(df_prophet_train)
        future = prophet_model.make_future_dataframe(periods=len(test_data), include_history=False, freq='D')
        prophet_forecast = prophet_model.predict(future)
        predictions = prophet_forecast['yhat'].values
        
    else:
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
            
    # Calcolo delle metriche
    predictions = np.maximum(0, predictions)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'MAE': mean_absolute_error(y_test, predictions),
        'R2': r2_score(y_test, predictions),
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
    
    test_days = st.slider("Numero di giorni da prevedere", 30, 90, 30)
    
    st.markdown("---")
    st.header("📊 Parametri Economici")
    unit_margin = st.number_input("Margine per Unità Venduta (€)", value=10.0, step=0.5)
    overstock_daily_cost = st.number_input("Costo Mantenimento Magazzino (€/unità)", value=0.5, step=0.1)

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
    train_end_date = df_single.index[-1] - pd.Timedelta(days=test_days)
    df_train_single = df_single.loc[:train_end_date].copy()
    df_test_single = df_single.loc[train_end_date + pd.Timedelta(days=1):].copy()

    # Feature Engineering
    df_with_features = create_features(df_single)
    features_to_use = [
        'giorno_della_settimana', 'settimana_dell_anno', 'mese', 'is_weekend', 
        'vendite_lag_1', 'vendite_lag_7', 'media_mobile_7g'
    ]
    
    df_train_with_features = df_with_features.loc[:train_end_date]
    df_test_with_features = df_with_features.loc[train_end_date + pd.Timedelta(days=1):]

    models = ["Holt-Winters", "ARIMA", "Prophet", "LightGBM", "XGBoost"]
    results = []
    predictions_dict = {}
    prophet_forecast_dict = {}

    # Esecuzione dei modelli
    with st.spinner("Addestramento e valutazione dei modelli..."):
        for model_name in models:
            if model_name in ["LightGBM", "XGBoost"]:
                metrics, predictions, _, prophet_forecast = train_and_evaluate_model(
                    model_name, df_train_with_features, df_test_with_features, features_to_use
                )
            else:
                metrics, predictions, _, prophet_forecast = train_and_evaluate_model(
                    model_name, df_train_single, df_test_single, []
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
        st.warning(f"Impossibile eseguire la decomposizione. Errore: {e}")
