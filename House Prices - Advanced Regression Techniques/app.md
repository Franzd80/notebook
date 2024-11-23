## Descrizione del Progetto di Predizione dei Prezzi delle Case

Il progetto consiste in un'applicazione web sviluppata utilizzando FastAPI, che consente di effettuare predizioni sui prezzi delle case basate su un modello di machine learning pre-addestrato. Questo sistema è progettato per accettare dati in formato JSON o CSV e restituire le stime dei prezzi delle abitazioni.

### Componenti Principali

1. **FastAPI**: Un framework moderno e veloce per costruire API web in Python, che facilita la creazione di endpoint per gestire richieste HTTP.

2. **Modello di Machine Learning**: Utilizza un modello salvato (in formato `.pkl`) che è stato addestrato su un dataset di prezzi delle case. Il modello è responsabile della generazione delle predizioni basate sulle caratteristiche fornite.

3. **StandardScaler**: Un preprocessore di scikit-learn utilizzato per normalizzare i dati in ingresso, assicurando che le feature abbiano una media di 0 e una deviazione standard di 1. Questo è fondamentale per migliorare le prestazioni del modello.

4. **Pydantic**: Utilizzato per definire e validare il formato dei dati in ingresso attraverso classi Python, garantendo che i dati ricevuti siano nel formato corretto e contengano tutte le informazioni necessarie.

### Funzionalità dell'Applicazione

#### 1. Predizione Singola
- **Endpoint**: `/predict`
- **Metodo**: POST
- **Input**: Un oggetto JSON che rappresenta le caratteristiche della casa (ad esempio, numero di stanze, anno di costruzione, qualità del tetto, ecc.).
- **Output**: Un oggetto JSON contenente la predizione del prezzo della casa.

#### 2. Predizione Multipla da CSV
- **Endpoint**: `/predict_csv`
- **Metodo**: POST
- **Input**: Un file CSV contenente più righe con le stesse caratteristiche richieste per la predizione.
- **Output**: Un oggetto JSON che restituisce una lista di predizioni per ogni riga nel CSV.

### Architettura del Progetto

- **Caricamento dei Modelli**: All'avvio dell'applicazione, il modello di machine learning e lo scaler vengono caricati dalla memoria.
- **Validazione dei Dati**: Prima di effettuare la predizione, l'app verifica che tutte le colonne richieste siano presenti nel file CSV o nell'oggetto JSON.
- **Scalabilità**: Grazie all'uso di FastAPI, l'app è progettata per gestire richieste simultanee in modo efficiente.

### Esempi di Utilizzo

- Un utente può inviare una richiesta POST a `/predict` con un payload JSON contenente i dettagli della casa e ricevere immediatamente una stima del prezzo.
- Gli utenti possono caricare un file CSV con più case per ottenere predizioni in batch, facilitando l'analisi dei dati immobiliari su larga scala.

### Conclusioni

Questo progetto rappresenta un'applicazione pratica dell'intelligenza artificiale nel settore immobiliare, consentendo agli utenti di ottenere valutazioni rapide e accurate dei prezzi delle case. La combinazione di FastAPI, machine learning e gestione dei dati tramite Pydantic rende il sistema robusto e facile da usare.
