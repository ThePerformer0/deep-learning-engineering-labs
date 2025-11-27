import os
import numpy as np
import json
from flask import Flask, request, jsonify
from tensorflow import keras

# --- 1. Initialisation de l'application Flask ---
app = Flask(__name__)

# --- 2. Chargement du modèle Keras ---
# On définit le chemin du modèle sauvegardé
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_model.h5")

# Le modèle est chargé une seule fois au démarrage du conteneur
try:
    # On désactive les logs TensorFlow excessifs lors du chargement
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modèle chargé avec succès depuis: {MODEL_PATH}")
except Exception as e:
    # Si le modèle n'est pas trouvé (erreur commune), on alerte
    print(f"ERREUR: Impossible de charger le modèle depuis {MODEL_PATH}. Levez l'erreur: {e}")
    model = None # Le conteneur peut démarrer, mais les requêtes échoueront.

# --- 3. Route d'état de santé (Health Check) ---
# Essentiel pour Docker et les services Cloud (K8s, Cloud Run)
@app.route("/health", methods=["GET"])
def health_check():
    """Vérifie que l'API est vivante et que le modèle est chargé."""
    if model:
        return jsonify({"status": "ok", "model_loaded": True}), 200
    else:
        return jsonify({"status": "error", "model_loaded": False}), 503

# --- 4. Route de prédiction ---
@app.route("/predict", methods=["POST"])
def predict():
    """Reçoit une image de 784 pixels, fait la prédiction et retourne le résultat."""
    
    # 4.1. Vérification du chargement du modèle
    if model is None:
        return jsonify({"error": "Model not available"}), 503
    
    # 4.2. Récupération des données d'entrée
    try:
        data = request.get_json(force=True)
        # On attend une clé 'image' contenant une liste de 784 flottants
        input_data = data["image"]
        
        # 4.3. Préparation des données pour Keras
        # Convertir la liste en array NumPy et s'assurer de la bonne forme (1, 784)
        input_array = np.array(input_data).astype("float32").reshape(1, -1)
        
        # Le modèle attend des valeurs normalisées (0.0 à 1.0)
        # S'assurer que les données sont déjà normalisées côté client ou ici si nécessaire
        
    except Exception as e:
        return jsonify({"error": f"Invalid input format: {e}"}), 400

    # 4.4. Prédiction
    try:
        predictions = model.predict(input_array)
        # Récupérer la classe prédite (l'indice de la probabilité maximale)
        predicted_class = int(np.argmax(predictions[0]))
        
        # Convertir le tableau NumPy de probabilités en liste Python pour JSON
        probabilities = predictions[0].tolist()

        # 4.5. Retour du résultat
        return jsonify({
            "prediction": predicted_class,
            "probabilities": probabilities,
            "confidence": f"{max(probabilities)*100:.2f}%"
        }), 200
        
    except Exception as e:
        # Erreur pendant l'inférence
        return jsonify({"error": f"Error during model prediction: {e}"}), 500

# --- 5. Démarrage (uniquement pour le développement local) ---
if __name__ == "__main__":
    # N.B. : En production (Docker/Gunicorn), c'est CMD qui démarre le serveur.
    app.run(host="0.0.0.0", port=5000, debug=True)