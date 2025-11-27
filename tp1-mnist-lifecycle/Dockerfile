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
# On copie tout le contenu du sous-dossier tp1-mnist-lifecycle/
# y compris src/ et models/ qui sont nécessaires à l'API.
COPY src/ /app/src/
COPY models/ /app/models/

# Exposer le port de l'application Flask
EXPOSE 5000

# Commande pour démarrer l'application en production avec Gunicorn (serveur WSGI)
# Gunicorn est un serveur HTTP plus robuste que le serveur Flask de dev.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.app:app"]