# --- ÉTAPE 1: Construction (Build Stage) ---
# On utilise une image Python plus complète pour installer les dépendances
FROM python:3.9-slim as builder

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# --- ÉTAPE 2: Image Finale (Production Stage) ---
# Utiliser une image Python slim pour minimiser la taille finale du conteneur
FROM python:3.9-slim

# Définir à nouveau le répertoire de travail
WORKDIR /app

# Copier les dépendances installées depuis l'étape 'builder'
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copier les fichiers de l'application
# On copie le code source du TP2
COPY src/ /app/src/
COPY models/ /app/models/

# Créer le répertoire mlruns pour MLflow (si nécessaire)
RUN mkdir -p /app/mlruns

# Définir la variable d'environnement pour MLflow
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# Commande par défaut : exécuter le script d'entraînement
# L'utilisateur peut override cette commande pour d'autres actions
CMD ["python", "src/train.py"]
