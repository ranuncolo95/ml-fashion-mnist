# importo librerie necessarie e modelli pretrainati
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # navigazione cartella di lavoro
    root_dir = os.getcwd()
    os.chdir(root_dir)                                  


    # settaggio delle immagini e del batch size
    batch_size = 64                         # Numero di immagini per batch che analizza per aggiornare i pesi del modello
    img_height = 28                         # Altezza delle immagini
    img_width = 28                         # Larghezza delle immagini

    tf.keras.backend.clear_session()        

    # Suddividisione dei dataset in train, validation e test
    (ds_train, ds_val, ds_test), ds_info = tfds.load('fashion_mnist',
    split=[
        'train',    # 100% del dataset train per il training
        'test[50%:]',     # primo 50% del dataset test per il validation
        'test[:50%]'      # secondo 50% del dataset test per il test
        ],
    #batch_size=batch_size,  # Imposta il batch size
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    #cache_dir=root_dir # questa parte lo scarica dentro la current directory
)
    class_names = ds_info.features['label'].names
    
    #
    # PREPARAZIONE DEL DATASET PER IL TRAINING
    #

    def normalize_image(img, label):
        return tf.cast(img, tf.float32) / 255., label   # Normalizza le immagini tra 0 e 1
    
    # data augmentation, opzionale

    '''
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"), # flipping orizzontale o verticale opzionale
        layers.RandomRotation(0.05),     # ±18° rotazione immagine
        layers.RandomZoom(0.1, 0.1),     # zoom (10% max)
        layers.RandomContrast(0.1),      # contrasto
        layers.RandomBrightness(0.1),    # lumininosità
        layers.RandomTranslation(        # spostamento
            height_factor=0.05,
            width_factor=0.05,
            fill_mode='reflect'          # riempimento dei bordi
        ),
    ])
    '''

    # Prepara il dataset di training: normalizza, memorizza in cache, mescola, crea batch e prefetch
    ds_train = ds_train.map(normalize_image, num_parallel_calls = tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Prepara il dataset di test: normalizza, crea batch, memorizza in cache e prefetch
    ds_val = ds_val.map(normalize_image, num_parallel_calls = tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)


    #
    # DEFINIZIONE DEL MODELLO
    #

    # Definisce il modello sequenziale: Flatten, Dense, Dropout, Dense finale
    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),      # Normalizza i pixel tra 0 e 1; input: immagini 28x28 RGB
        tf.keras.layers.Conv2D(64, 5, activation='relu'),                               # Convoluzione: 32 filtri, kernel 5x5, attivazione ReLU
        tf.keras.layers.MaxPooling2D(),                                                 # Pooling massimo 2x2 (default)
        tf.keras.layers.Dropout(0.2),                                                   # Dropout: disattiva il 10% dei neuroni (riduce overfitting)
        tf.keras.layers.Conv2D(32, 5, activation='relu'),                             # Seconda convoluzione: 16 filtri, kernel 5x5, ReLU
        tf.keras.layers.MaxPooling2D(),                                               # Altro pooling massimo 2x2
        tf.keras.layers.Dropout(0.1),                                                 # Dropout: disattiva il 20% dei neuroni
        tf.keras.layers.Flatten(),                                                      # Appiattisce i dati in un vettore 1D
        tf.keras.layers.Dense(64, activation='relu'),                                   # Strato denso: 32 neuroni, attivazione ReLU
        tf.keras.layers.Dense(len(class_names), activation='softmax')                   # Output: un neurone per classe, softmax per probabilità
    ])

    model.summary()   # Mostra un riepilogo del modello


    #
    # COMPILAZIONE DEL MODELLO
    #

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)  # Funzione di perdita

    # Compila il modello con ottimizzatore Adam, funzione di perdita e accuratezza come metrica
    model.compile(
        optimizer = 'adam',
        loss = loss_fn,
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    #
    # ALLENAMENTO DEL MODELLO, ULTERIORI SETTAGGI DURANTE LE EPOCHE
    #

    tensorboard_callback = TensorBoard()  # Callback per TensorBoard

    class TrainingMonitor(Callback):
        def __init__(self, patience=3):
            super().__init__()
            self.patience = patience
            self.best_weights = None
        
        def on_train_begin(self, logs=None):
            # Initialize tracking variables
            self.val_losses = []
            self.train_losses = []
            self.wait = 0  # For early stopping
            
        def on_epoch_end(self, epoch, logs=None):
            # 1. Record metrics
            current_val_loss = logs.get('val_loss', np.inf)
            current_train_loss = logs.get('loss', np.inf)
            self.val_losses.append(current_val_loss)
            self.train_losses.append(current_train_loss)
            
            # 2. Overfitting detection
            if epoch > 0:
                if (current_val_loss > self.val_losses[-2] and 
                    current_train_loss < self.train_losses[-2]):
                    print(f"\n⚠️ Overfitting alert! (Epoch {epoch+1})")
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.model.stop_training = True
                        print("Stopping training due to persistent overfitting")
                else:
                    self.wait = 0
            
            # 3. Underfitting check
            if logs.get('val_accuracy', 0) < 0.6 and epoch > 5:
                print("\n⚠️ Underfitting detected - consider model changes")
            
            # 4. Save best weights
            if current_val_loss == np.min(self.val_losses):
                self.best_weights = self.model.get_weights()
                
    # Addestra il modello
    early_stopping = EarlyStopping(
        monitor ='sparse_categorical_accuracy',         # Monitora l'accuratezza di validazione
        patience = 5,                                   # Numero di epoche senza miglioramento prima di fermare l'addestramento
        restore_best_weights=True,                      # Ripristina i pesi del modello alla migliore versione trovata
        mode = 'max',                                   # Modalità di monitoraggio: 'max' per l'accuratezza
    )


    callbacks = [early_stopping, TrainingMonitor(), tensorboard_callback]

    history = model.fit(
        ds_train,                         # Dati di addestramento
        epochs = 15,                       # Numero di epoche per l'addestramento
        validation_data=ds_val,           # Dati di validazione
        callbacks = callbacks,            # Callback per il monitoraggio durante le epoche
        shuffle=False                     # Non mescola i dati durante l'addestramento
    )

    model.evaluate(ds_val, verbose = 2)   # Valuta il modello sul dataset di test
    model.save("analisi_fashion_2.keras")   # Salva il modello addestrato

    #
    # VISUALIZZAZIONE DEI RISULTATI DELL'ALLENAMENTO
    #
    
    # Crazione del dataframe con le metriche di allenamento
    history_df = pd.DataFrame(history.history)
    print(history_df.columns)  # Mostra le colonne del dataframe
    
    
    # Creazione dei grafici
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plottiamo l'andamento della funzione di perdita (Loss) nel primo subplot
    history_df[["loss", "val_loss"]].plot(
        ax=ax1,
        color=['blue', 'orange']
    )
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # Plottiamo l'andamento dell'accuratezza (Accuracy) della previsione nel secondo subplot'
    history_df[["sparse_categorical_accuracy", "val_sparse_categorical_accuracy"]].plot(
        ax=ax2,
        color=['green', 'red']
    )
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
