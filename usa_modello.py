import tensorflow as tf                   # Importa TensorFlow
import numpy as np                        # Importa NumPy per operazioni numeriche
import matplotlib.pyplot as plt           # Importa Matplotlib per visualizzare immagini
import tensorflow_datasets as tfds
import os

path = os.path.join(os.getcwd())
os.chdir(path)

model_path = os.path.join(os.getcwd(), "analisi_fashion.keras")    # Percorso del modello salvato
model = tf.keras.models.load_model(model_path)           # Carica il modello addestrato

# Caricamento del dataset suddiviso
(ds_train, ds_val, ds_test), ds_info = tfds.load('fashion_mnist',
    split=[
        'train',          # 100% del dataset train per il training
        'test[50%:]',     # primo 50% del dataset test per il validation
        'test[:50%]'      # secondo 50% del dataset test per il test
        ],
    #batch_size=batch_size,  # Imposta il batch size
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    #cache_dir=root_dir # questa parte lo scarica dentro la current directory
)

# Recupera i nomi delle classi
class_names = ds_info.features['label'].names

# Pre-elabora il test set (batch, normalizzazione, ecc.)
test_data = ds_test.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
test_data = test_data.batch(25).prefetch(tf.data.AUTOTUNE)

# Visualizzazione delle predizioni
for images, labels in test_data.take(1):
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 12))
    for i in range(min(25, len(images))):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy())
        true_label = class_names[labels[i].numpy()]
        predicted_label = class_names[predicted_labels[i]]
        color = "green" if predicted_label == true_label else "red"
        plt.title(f"Pred: {predicted_label}\n(True: {true_label})", color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    break