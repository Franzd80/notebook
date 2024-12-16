Descrizione del Notebook
1. Importazione delle Librerie
Il notebook inizia importando le librerie necessarie per l'implementazione, tra cui TensorFlow, Keras, Pandas, NumPy e scikit-learn. Queste librerie sono fondamentali per costruire e addestrare modelli di machine learning.

2. Caricamento e Preprocessing dei Dati
Caricamento del Dataset: I dati vengono caricati da un file CSV utilizzando pandas. Il dataset sembra contenere tweet con etichette di classificazione.
Selezione delle Colonne: Vengono selezionate solo le colonne rilevanti (etichetta e testo).
Divisione dei Dati: I dati vengono divisi in variabili di input (x) e output (y).
3. Implementazione dei Layer Custom
a. TransformerBlock
Questo layer implementa un blocco Transformer utilizzando l'attenzione multi-testa e un feed-forward network. Include anche normalizzazione e dropout per migliorare la generalizzazione del modello.
b. TokenAndPositionEmbedding
Questo layer gestisce l'embedding dei token e delle posizioni. Combina gli embeddings delle parole con le informazioni posizionali necessarie per il modello Transformer.
4. Creazione del Dataset
Tokenizzazione: Viene creato un tokenizer per convertire il testo in sequenze numeriche.
Padding: Le sequenze vengono uniformate a una lunghezza massima di 200 parole.
5. Costruzione del Modello
Viene definito un modello di rete neurale che utilizza i layer custom precedentemente definiti. Il modello include:
Un layer di embedding per i token e le posizioni.
Un blocco Transformer.
Un layer di pooling globale.
Dense layers per la classificazione finale.
6. Compilazione e Addestramento del Modello
Il modello viene compilato utilizzando l'ottimizzatore Adam e la funzione di perdita sparse_categorical_crossentropy.
Viene avviato il processo di addestramento con 5 epoche, e viene monitorata l'accuratezza sia sui dati di addestramento che su quelli di validazione.
7. Valutazione del Modello
Dopo l'addestramento, il modello viene valutato sui dati di test per misurare le sue performance.
8. Salvataggio e Caricamento del Modello
Il modello addestrato viene salvato su disco e successivamente caricato per dimostrare che i pesi possono essere recuperati correttamente.
