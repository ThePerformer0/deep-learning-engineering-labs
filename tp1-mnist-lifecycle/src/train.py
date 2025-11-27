import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Configuration pour éviter les logs excessifs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data():
    """Charge et normalise les données MNIST."""
    print("Chargement des données MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalisation (0-255 -> 0.0-1.0)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Redimensionnement
    x_train = x_train.reshape((60000, 784))
    x_test = x_test.reshape((10000, 784))
    
    return (x_train, y_train), (x_test, y_test)

def build_model(input_shape=(784,), num_classes=10):
    """Construit le modèle séquentiel Keras."""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # 1. Préparation des données
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # 2. Construction du modèle
    model = build_model()
    model.summary()

    # 3. Entraînement
    print("\nDébut de l'entraînement...")
    history = model.fit(
        x_train, 
        y_train, 
        epochs=5, 
        batch_size=128, 
        validation_split=0.1
    )

    # 4. Évaluation
    print("\nÉvaluation sur le jeu de test...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Précision sur les données de test: {test_acc:.4f}")

    # 5. Sauvegarde
    # On sauvegarde dans le dossier 'models'
    model_path = os.path.join("..", "models", "mnist_model.h5")
    model.save(model_path)
    print(f"Modèle sauvegardé sous {model_path}")

if __name__ == "__main__":
    main()