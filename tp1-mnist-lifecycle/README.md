# 🧪 TP1 : Cycle de Vie et Déploiement d'un Modèle de Deep Learning (MNIST)

Ce dépôt contient le premier Travail Pratique de la série "Deep Learning Engineering Labs" axée sur la mise en œuvre des principes du MLOps. Il couvre l'intégralité du cycle de vie d'un modèle, de l'entraînement à la conteneurisation en vue du déploiement.

---

## 🚀 Objectifs et Compétences Acquises

L'objectif principal est de transformer un script d'entraînement académique en un service professionnel prêt pour la production.

1.  **Modélisation DL :** Construction et entraînement d'un réseau de neurones dense (MLP) pour la classification des chiffres manuscrits MNIST avec Keras/TensorFlow.
2.  **MLOps & Traçabilité :** Intégration de **MLflow** pour le suivi des hyperparamètres, des métriques et l'archivage du modèle.
3.  **Conteneurisation :** Création d'une API d'inférence avec **Flask/Gunicorn** et empaquetage dans une image **Docker** optimisée.
4.  **Ingénierie Logicielle :** Structuration du projet, gestion des dépendances (`requirements.txt`) et versionnement (`.gitignore`).

---

## 🛠️ Stack Technique

| Catégorie | Outil | Version | Description |
| :--- | :--- | :--- | :--- |
| **Framework DL** | TensorFlow / Keras | 2.x | Construction, entraînement et sérialisation du modèle. |
| **MLOps** | MLflow | Récent | Traçage des expérimentations pour la reproductibilité. |
| **API Web** | Flask / Gunicorn | Récent | Serveur WSGI pour exposer le modèle via une API REST. |
| **Conteneurisation** | Docker | Récent | Environnement isolé pour le déploiement. |
| **Langage** | Python | 3.9+ | Langage de développement principal. |

---

## ⚙️ Structure du Projet

Le projet est structuré selon les bonnes pratiques MLOps pour séparer le code source, les modèles et la documentation.

```text
tp1-mnist-lifecycle/
├── src/
│   ├── train.py      # Script d'entraînement, incluant le logging MLflow.
│   └── app.py        # API Flask pour le service d'inférence.
├── models/           # Répertoire de sortie pour le modèle sauvegardé (mnist_model.h5).
├── report/           # Rapport du TP au format LaTeX (main.tex).
├── Dockerfile        # Fichier d'instructions pour construire le conteneur.
└── requirements.txt  # Dépendances Python nécessaires au projet.
````

-----

## 🚀 Instructions d'Exécution

### Prérequis

  * [Git](https://git-scm.com/)
  * [Python 3.9+](https://www.python.org/downloads/)
  * [Docker](https://www.docker.com/get-started) (Doit être en cours d'exécution)

### 1\. Configuration et Entraînement

**NOTE :** Assurez-vous que l'environnement virtuel est activé et que les dépendances sont installées (voir le `requirements.txt` à la racine du monorepo).

```bash
# Se placer dans le répertoire du TP
cd deep-learning-engineering-labs/tp1-mnist-lifecycle

# Entraîner le modèle et logguer l'expérimentation
python src/train.py
```

### 2\. Suivi des Expérimentations (MLflow)

Après l'exécution, les résultats sont stockés dans le dossier `mlruns/`. Lancez l'interface utilisateur pour visualiser l'expérience :

```bash
# Remonter à la racine du monorepo (où se trouve le dossier mlruns/)
cd .. 

# Lancer le serveur MLflow
mlflow ui

# Accédez à l'interface via votre navigateur à [http://127.0.0.1:5000](http://127.0.0.1:5000)
```

### 3\. Conteneurisation (Docker)

La construction de l'image utilise une approche multi-stage pour minimiser la taille finale du conteneur.

```bash
# S'assurer d'être dans le dossier tp1-mnist-lifecycle/
docker build -t mnist-api:latest .
```

### 4\. Démarrage et Test de l'API

Démarrez le conteneur en mappant le port 5000 du conteneur sur le port 5000 de la machine hôte.

```bash
# Lancement du conteneur en arrière-plan (-d)
docker run -d -p 5000:5000 --name mnist-service mnist-api:latest

# --- Test de l'état de santé (Health Check) ---
# Vérifie que l'API est en ligne et que le modèle est chargé
curl http://localhost:5000/health
# Résultat attendu : {"model_loaded": true, "status": "ok"}
```

#### Test d'Inférence (Exemple)

Pour tester la prédiction, vous devez envoyer un array de 784 valeurs (pixels normalisés, 0-1) d'une image MNIST.

```bash
# Exemple de corps de requête JSON (Image d'un "0" ou d'un "1" très simplifié)
# Vous devrez utiliser un exemple réel pour un test valide.
REQUEST_BODY='{"image": [0.0, 0.0, ..., 1.0, 1.0, 0.0, ...]}' 

curl -X POST \
  -H "Content-Type: application/json" \
  -d "$REQUEST_BODY" \
  http://localhost:5000/predict

# Résultat attendu : {"prediction": 7, "confidence": "99.98%", "probabilities": [...]}
```
