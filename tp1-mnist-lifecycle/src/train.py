import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import mlflow
import mlflow.keras

# Configuration pour éviter les logs excessifs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Définition des Hyperparamètres ---
# On les centralise pour les logguer facilement avec MLflow
HYPERPARAMS = {
    "epochs": 5,
    "batch_size": 128,
    "optimizer": "adam",
    "dropout_rate": 0.2,
    "num_classes": 10,
    "input_shape": 784
}

def load_data():
    """Charge et normalise les données MNIST."""
    print("Chargement des données MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalisation (0-255 -> 0.0-1.0)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Redimensionnement (Flattening)
    x_train = x_train.reshape((len(x_train), HYPERPARAMS["input_shape"]))
    x_test = x_test.reshape((len(x_test), HYPERPARAMS["input_shape"]))
    
    return (x_train, y_train), (x_test, y_test)

def build_model(params):
    """Construit le modèle séquentiel Keras."""
    model = keras.Sequential([
        keras.layers.Input(shape=(params["input_shape"],)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(params["dropout_rate"]),
        keras.layers.Dense(params["num_classes"], activation='softmax')
    ])
    
    model.compile(
        optimizer=params["optimizer"],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # --- MLflow : Configuration de l'Expérience ---
    mlflow.set_experiment("TP1_MNIST_Deep_Learning_LifeCycle")
    
    # Démarre une nouvelle 'Run' (exécution) MLflow
    with mlflow.start_run(run_name="Run_Baseline_Dense_Model") as run:
        
        # 1. Préparation des données
        (x_train, y_train), (x_test, y_test) = load_data()
        
        # 2. Construction du modèle
        model = build_model(HYPERPARAMS)
        model.summary()

        # --- MLflow : Enregistrement des Hyperparamètres ---
        # Log des paramètres
        mlflow.log_params(HYPERPARAMS)

        # 3. Entraînement
        print("\nDébut de l'entraînement...")
        history = model.fit(
            x_train, 
            y_train, 
            epochs=HYPERPARAMS["epochs"], 
            batch_size=HYPERPARAMS["batch_size"], 
            validation_split=0.1,
            verbose=2 # Affiche moins de logs pendant l'entraînement
        )

        # 4. Évaluation
        print("\nÉvaluation sur le jeu de test...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        print(f"Précision sur les données de test: {test_acc:.4f}")
        
        # --- MLflow : Enregistrement des Métriques ---
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        
        # --- MLflow : Enregistrement du Modèle (Format Keras) ---
        # Cela enregistre le modèle dans le répertoire MLruns,
        # le rendant récupérable pour l'API Flask plus tard.
        mlflow.keras.log_model(model, "model")
        
        # Le modèle est également sauvegardé localement dans le dossier models/
        model_path = os.path.join("..", "models", "mnist_model.h5")
        # Keras .save() est toujours utile pour un accès direct si MLflow n'est pas utilisé
        model.save(model_path)
        print(f"Modèle sauvegardé localement sous {model_path}")

if __name__ == "__main__":
    main()