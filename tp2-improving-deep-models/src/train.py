import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import mlflow
import mlflow.keras

# Configuration pour éviter les logs excessifs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Définition des Hyperparamètres ---
# Ajout des nouveaux hyperparamètres pour les améliorations
HYPERPARAMS = {
    "epochs": 20,  # Augmenté pour permettre l'early stopping
    "batch_size": 128,
    "optimizer": "adam",
    "dropout_rate": 0.3,  # Légèrement augmenté
    "num_classes": 10,
    "input_shape": 784,
    "l2_lambda": 0.001,  # Nouveau : coefficient de régularisation L2
    "use_batch_norm": True,  # Nouveau : utiliser Batch Normalization
    "early_stopping_patience": 5,  # Nouveau : patience pour early stopping
    "early_stopping_monitor": "val_loss"  # Nouveau : métrique à surveiller
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
    """
    Construit le modèle séquentiel Keras avec améliorations :
    - Régularisation L2 sur les poids
    - Batch Normalization pour stabiliser l'entraînement
    - Dropout pour réduire le surapprentissage
    """
    model = keras.Sequential()
    
    # Couche d'entrée
    model.add(keras.layers.Input(shape=(params["input_shape"],)))
    
    # Première couche Dense avec régularisation L2
    model.add(keras.layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=regularizers.l2(params["l2_lambda"])
    ))
    
    # Batch Normalization (si activé)
    if params["use_batch_norm"]:
        model.add(keras.layers.BatchNormalization())
    
    # Dropout
    model.add(keras.layers.Dropout(params["dropout_rate"]))
    
    # Couche de sortie avec régularisation L2
    model.add(keras.layers.Dense(
        params["num_classes"], 
        activation='softmax',
        kernel_regularizer=regularizers.l2(params["l2_lambda"])
    ))
    
    model.compile(
        optimizer=params["optimizer"],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # --- MLflow : Configuration de l'Expérience ---
    mlflow.set_experiment("TP2_Improving_Deep_Models")
    
    # Démarre une nouvelle 'Run' (exécution) MLflow
    with mlflow.start_run(run_name="Run_Improved_Model_L2_BN_ES") as run:
        
        # 1. Préparation des données
        (x_train, y_train), (x_test, y_test) = load_data()
        
        # 2. Construction du modèle amélioré
        model = build_model(HYPERPARAMS)
        model.summary()

        # --- MLflow : Enregistrement des Hyperparamètres ---
        # Log des paramètres (incluant les nouveaux)
        mlflow.log_params(HYPERPARAMS)
        
        # Log d'un flag pour indiquer que la régularisation est appliquée
        mlflow.log_param("regularization_applied", True)

        # 3. Configuration des callbacks
        callbacks = []
        
        # Early Stopping pour éviter le surapprentissage
        early_stopping = EarlyStopping(
            monitor=HYPERPARAMS["early_stopping_monitor"],
            patience=HYPERPARAMS["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # 4. Entraînement avec validation
        print("\nDébut de l'entraînement avec améliorations...")
        print(f"Techniques appliquées:")
        print(f"  - Régularisation L2 (lambda={HYPERPARAMS['l2_lambda']})")
        print(f"  - Batch Normalization: {HYPERPARAMS['use_batch_norm']}")
        print(f"  - Early Stopping (patience={HYPERPARAMS['early_stopping_patience']})")
        print(f"  - Dropout (rate={HYPERPARAMS['dropout_rate']})")
        
        history = model.fit(
            x_train, 
            y_train, 
            epochs=HYPERPARAMS["epochs"], 
            batch_size=HYPERPARAMS["batch_size"], 
            validation_split=0.1,
            callbacks=callbacks,
            verbose=2
        )

        # 5. Évaluation sur le jeu de test
        print("\nÉvaluation sur le jeu de test...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        print(f"Précision finale sur les données de test: {test_acc:.4f}")
        print(f"Perte finale sur les données de test: {test_loss:.4f}")
        
        # Récupération des meilleures métriques de validation depuis l'historique
        best_val_loss = min(history.history['val_loss'])
        best_val_acc = max(history.history['val_accuracy'])
        final_val_loss = history.history['val_loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        print(f"\nMeilleures métriques de validation:")
        print(f"  - Meilleure précision: {best_val_acc:.4f}")
        print(f"  - Meilleure perte: {best_val_loss:.4f}")
        print(f"\nMétriques finales de validation:")
        print(f"  - Précision finale: {final_val_acc:.4f}")
        print(f"  - Perte finale: {final_val_loss:.4f}")
        
        # --- MLflow : Enregistrement des Métriques ---
        # Métriques de test
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_loss", test_loss)
        
        # Métriques de validation
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_metric("best_val_loss", best_val_loss)
        
        # Nombre d'epochs réellement utilisés (peut être inférieur à epochs max si early stopping)
        actual_epochs = len(history.history['loss'])
        mlflow.log_metric("actual_epochs", actual_epochs)
        mlflow.log_param("actual_epochs", actual_epochs)
        
        # --- MLflow : Enregistrement du Modèle (Format Keras) ---
        mlflow.keras.log_model(model, "model")
        
        # Le modèle est également sauvegardé localement dans le dossier models/
        model_path = os.path.join("..", "models", "mnist_improved_model.h5")
        model.save(model_path)
        print(f"\nModèle sauvegardé localement sous {model_path}")

if __name__ == "__main__":
    main()

