# Project work proposal

> Sent 2026-10-14 to andrea.asperti@unibo.it (Italian original, verbatim).
> Subject area: Project Work in Deep Learning, 3 CFU, prof. Andrea Asperti.

---

Buongiorno Prof. Asperti,

sono Klein Tahiraj, studente della laurea magistrale in Artificial Intelligence. Sono fuori corso da quattro anni che ho passato lavorando per diverse startup; ora mi sono preso una pausa dal lavoro per chiudere definitivamente il percorso. Le scrivo per il project work in deep learning, perché oltre alla tesi, che sto scrivendo, mi manca solo quello. Punto a laurearmi a inizio dicembre, quindi dovrei che dovrei avere convalidato tutti i crediti entro il 13 novembre.

Come tesi sto facendo un lavoro sull'interpretabilità meccanicistica di transformer addestrati a seguire lo stato del mondo in Blocksworld (5 blocchi, stato descritto da 36 fatti binari come on(A,B), clear(C), on_table(D), legati da vincoli logici). Per farla breve: i probe lineari leggono lo stato dalle attivazioni quasi perfettamente, ma intervenire lungo quelle stesse direzioni in genere non cambia il comportamento del modello, e la tesi studia il perché.

Negli ultimi mesi ho fatto molti esperimenti, e tra i rami che ho visto emergere ce n'è uno che ho abbandonato abbastanza presto (quindi non entrerà nella tesi), ma che mi piacerebbe approfondire come project work: in pratica ho notato che uno Sparse Autoencoder addestrato sulle attivazioni trova, o non trova, feature legate ai fatti dello stato a seconda del modello: su un Qwen-0.5B fine-tuned sullo stesso task nessuna, sui transformer piccoli molte, ma solo debolmente correlate. Un bottleneck supervisionato invece recupera una struttura pulita, ma solo perché è la supervisione a imporla. La domanda che resta aperta è se un modello a variabili latenti non supervisionato recuperi i fattori veri, uno a uno.

Controllando un po' la letteratura sul tema ho visto il suo lavoro "Sparsity in Variational Autoencoders", che mi ha ispirato questa direzione. Da quanto ho capito, in spazi latenti abbastanza grandi un VAE disattiva una parte delle variabili latenti, che il generatore ignora del tutto, e per una data architettura sembra esistere una dimensione interna intrinseca dei dati. Su MNIST, per quanto ho visto, il numero di fattori veri non è noto; in Blocksworld invece il numero di stati legali e di fattori indipendenti si conosce esattamente. Mi piacerebbe quindi vedere se il numero di variabili attive segue il numero di fattori veri, e quanto dipende dall'architettura e da beta. In più, i risultati teorici che legano variabili attive e dimensione della varietà (Dai & Wipf 2019, Zheng et al. 2022) assumono dati continui e un decoder gaussiano, mentre qui i dati sono binari.

Quindi pensavo di esplorare un po' queste direzioni:
1. Fissare in anticipo il "numero giusto" di fattori (su dati discreti la dimensione non è univoca: bit di informazione, fattori minimi, stima di dimensione intrinseca).
2. Addestrare VAE con spazio latente sovradimensionato, sugli stati e sulle attivazioni dei transformer, variando beta e iperparametri, e contare quante variabili restano attive rispetto al numero atteso.
3. Verificare se ogni variabile attiva corrisponde a un singolo fatto, confrontando beta-VAE, beta-TCVAE, un VAE a latenti discrete e lo SAE.
4. Generare stati dal VAE e misurare quanti rispettano i vincoli del dominio.

Cosa ne pensa? Le sembra adatto come project work? Mi faccia sapere!

Grazie, un saluto,

Klein Tahiraj
