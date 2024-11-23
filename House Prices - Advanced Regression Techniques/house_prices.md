Il progetto  riguarda l'analisi e la modellizzazione dei dati relativi ai prezzi delle case utilizzando un algoritmo di regressione, in particolare il **Gradient Boosting Regressor**. Di seguito sono delineati i passaggi principali e le tecniche utilizzate nel progetto:

## Fasi del Progetto

### 1. Esplorazione dei Dati
- **Analisi Iniziale**: Comprensione della struttura del dataset, inclusi tipi di dati e statistiche descrittive.
- **Verifica dei Valori Mancanti**: Identificazione delle colonne con dati mancanti per una successiva gestione.

### 2. Visualizzazione dei Dati
- **Grafici**: Creazione di istogrammi, boxplot e heatmap di correlazione per visualizzare la distribuzione delle variabili e le relazioni tra di esse.

### 3. Pulizia dei Dati
- **Rimozione di Colonne Non Necessarie**: Eliminazione di colonne che non contribuiscono all'analisi.
- **Gestione delle Anomalie**: Trattamento dei valori nulli e anomalie nei dati.

### 4. Scaling delle Caratteristiche
- **Normalizzazione**: Applicazione di tecniche di scaling per le variabili numeriche al fine di migliorare le prestazioni del modello.

### 5. Modellizzazione
- **Utilizzo di Algoritmi di Regressione**: Implementazione del Gradient Boosting Regressor per effettuare predizioni sui prezzi delle case.

### 6. Valutazione del Modello
- **Metriche di Valutazione**: Utilizzo di metriche come l'errore quadratico medio (MSE) e il coefficiente di determinazione (R²) per valutare le prestazioni del modello.

### Flusso di Lavoro
1. Caricamento dei dataset da file CSV.
2. Analisi preliminare con `df.info()` e `df.describe()` per comprendere la distribuzione e i tipi di dati.
3. Creazione di grafici per visualizzare la distribuzione delle variabili categoriali e numeriche.
4. Pulizia dei dati, inclusa la gestione dei valori mancanti tramite imputazione o sostituzione con valori appropriati.
5. Codifica delle variabili categoriche utilizzando `LabelEncoder`.
6. Applicazione del modello Gradient Boosting Regressor sui dati puliti e normalizzati.
7. Esportazione dei modelli per l'utilizzo su fastapi 
