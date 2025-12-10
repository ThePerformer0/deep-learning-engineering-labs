# Image légère Python
FROM python:3.9-slim

WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY src/ /app/src/
COPY report/ /app/report/

# Dossiers pour modèles et logs
RUN mkdir -p /app/models /app/mlruns

# Commande par défaut : entraînement rapide U-Net sur données simulées
CMD ["python", "src/unet_segmentation.py", "--epochs", "3", "--batch-size", "8"]

