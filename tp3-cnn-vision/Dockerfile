# Image de base légère
FROM python:3.9-slim

WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY src/ /app/src/

# Dossier pour modèles/outputs éventuels
RUN mkdir -p /app/models

# Commande par défaut : entraînement CNN basique
CMD ["python", "src/cnn_classification.py"]

