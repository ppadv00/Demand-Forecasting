# 📈 Demand Forecasting Dashboard

Questo repository contiene il codice per una dashboard di Demand Forecasting, sviluppata con Python e la libreria Streamlit. L'obiettivo è confrontare le performance di diversi modelli di Machine Learning e statistici nella previsione della domanda di prodotti, valutandoli sia dal punto di vista statistico che da quello economico.

## Caratteristiche Principali

-   **Generazione di Dati Simulati:** Utilizza un generatore di dati per simulare serie storiche di vendite con trend, stagionalità ed eventi speciali.
-   **Feature Engineering:** Creazione di feature temporali e statistiche (lag, medie mobili) per i modelli di Machine Learning.
-   **Confronto Modelli:** Valutazione di modelli come Holt-Winters, ARIMA, Prophet, LightGBM e XGBoost.
-   **Analisi Economica:** Calcolo dell'impatto finanziario degli errori di previsione (costi di understock e overstock).
-   **Visualizzazione Avanzata:** Grafici interattivi per visualizzare le previsioni e le componenti della serie storica.

## Come eseguire il progetto in locale

1.  Clona il repository:
    ```bash
    git clone [https://github.com/IlTuoUsername/IlTuoProgetto.git](https://github.com/IlTuoUsername/IlTuoProgetto.git)
    cd IlTuoProgetto
    ```
2.  Crea e attiva un ambiente virtuale (consigliato):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Per macOS/Linux
    venv\Scripts\activate      # Per Windows
    ```
3.  Installa le dipendenze:
    ```bash
    pip install -r requirements.txt
    ```
4.  Avvia la dashboard Streamlit:
    ```bash
    streamlit run app.py
    ```

## Deploy su Streamlit Community Cloud

Questo progetto è configurato per un deploy semplice e veloce su [Streamlit Community Cloud](https://streamlit.io/cloud). Carica il repository su GitHub e segui le istruzioni della piattaforma per collegarlo e pubblicarlo.
