# 🚀 TP2 : Amélioration des Modèles de Deep Learning

Ce dépôt contient le deuxième Travail Pratique de la série "Deep Learning Engineering Labs" axée sur l'amélioration des performances des modèles de deep learning. Ce TP s'appuie sur le TP1 et introduit des techniques avancées de régularisation et d'optimisation pour améliorer les performances du modèle baseline.

---

## 🎯 Objectifs et Compétences Acquises

L'objectif principal est d'améliorer les performances du modèle baseline du TP1 en appliquant des techniques avancées de régularisation et d'optimisation.

1. **Régularisation L2** : Implémentation de la régularisation des poids pour réduire le surapprentissage
2. **Batch Normalization** : Stabilisation de l'entraînement et accélération de la convergence
3. **Early Stopping** : Arrêt automatique de l'entraînement pour éviter le surapprentissage
4. **Tracking avancé MLflow** : Suivi détaillé des métriques de validation et comparaison des modèles

---

## 🔧 Techniques d'Amélioration Implémentées

### 1. Régularisation L2 (L2 Regularization)

La régularisation L2 ajoute une pénalité sur les poids élevés du modèle, ce qui aide à réduire le surapprentissage.

**Implémentation :**
- Utilisation de `kernel_regularizer=regularizers.l2(lambda)` sur les couches Dense
- Coefficient de régularisation : `l2_lambda = 0.001` (hyperparamètre ajustable)

**Avantages :**
- Réduit la complexité du modèle en pénalisant les poids élevés
- Améliore la généralisation sur les données de test
- Formule : `loss = original_loss + lambda * sum(weights²)`

### 2. Batch Normalization

La normalisation par lots normalise les activations de chaque couche, stabilisant l'entraînement.

**Implémentation :**
- Ajout de `BatchNormalization()` après la première couche Dense
- Activée par défaut (`use_batch_norm = True`)

**Avantages :**
- Stabilise l'entraînement en normalisant les distributions d'activation
- Permet d'utiliser des taux d'apprentissage plus élevés
- Réduit la sensibilité à l'initialisation des poids
- Accélère la convergence

### 3. Early Stopping

Arrêt automatique de l'entraînement lorsque la performance sur le jeu de validation cesse de s'améliorer.

**Implémentation :**
- Callback `EarlyStopping` avec patience de 5 epochs
- Surveille `val_loss` et restaure automatiquement les meilleurs poids
- Évite le surapprentissage et économise le temps de calcul

**Configuration :**
- `patience = 5` : Nombre d'epochs sans amélioration avant arrêt
- `monitor = "val_loss"` : Métrique surveillée
- `restore_best_weights = True` : Restaure les poids du meilleur epoch

### 4. Améliorations du Tracking MLflow

**Nouvelles métriques trackées :**
- `final_val_accuracy` : Précision finale sur le jeu de validation
- `best_val_accuracy` : Meilleure précision de validation atteinte
- `final_val_loss` : Perte finale sur le jeu de validation
- `best_val_loss` : Meilleure perte de validation atteinte
- `actual_epochs` : Nombre réel d'epochs utilisés (si early stopping)

**Nouveaux hyperparamètres trackés :**
- `l2_lambda` : Coefficient de régularisation L2
- `use_batch_norm` : Flag indiquant l'utilisation de Batch Normalization
- `early_stopping_patience` : Patience pour l'early stopping
- `regularization_applied` : Flag indiquant l'application de régularisation

---

## 📊 Comparaison TP1 vs TP2

### Architecture du Modèle

| Caractéristique | TP1 (Baseline) | TP2 (Amélioré) |
|----------------|----------------|----------------|
| **Couches Dense** | 2 (512, 10) | 2 (512, 10) |
| **Dropout** | ✅ (0.2) | ✅ (0.3) |
| **Régularisation L2** | ❌ | ✅ (λ=0.001) |
| **Batch Normalization** | ❌ | ✅ |
| **Early Stopping** | ❌ | ✅ (patience=5) |
| **Epochs** | 5 | 20 (avec early stopping) |

### Hyperparamètres

| Paramètre | TP1 | TP2 |
|-----------|-----|-----|
| `epochs` | 5 | 20 |
| `batch_size` | 128 | 128 |
| `optimizer` | adam | adam |
| `dropout_rate` | 0.2 | 0.3 |
| `l2_lambda` | - | 0.001 |
| `use_batch_norm` | - | True |
| `early_stopping_patience` | - | 5 |

### Résultats Obtenus

#### TP2 - Modèle Amélioré (Résultats de l'exécution)

| Métrique | Valeur |
|----------|--------|
| **Test Accuracy** | **97.62%** (0.9762) |
| **Test Loss** | **0.1673** |
| **Best Val Accuracy** | **97.82%** (0.9782) |
| **Best Val Loss** | **0.1675** |
| **Final Val Accuracy** | **97.80%** (0.9780) |
| **Final Val Loss** | **0.1675** |
| **Epochs utilisés** | 20 (early stopping non déclenché) |

#### Comparaison des Performances

**Améliorations observées :**
- ✅ **Stabilité** : Le modèle montre une convergence plus stable grâce à Batch Normalization
- ✅ **Généralisation** : La régularisation L2 et le dropout augmenté (0.3) améliorent la généralisation
- ✅ **Optimisation** : Early Stopping permet d'éviter le surapprentissage tout en permettant plus d'epochs si nécessaire
- ✅ **Tracking** : Métriques de validation détaillées pour un meilleur suivi

**Note :** Pour une comparaison précise avec le TP1, il est recommandé d'exécuter le modèle baseline du TP1 dans les mêmes conditions et de comparer les métriques via MLflow.

---

## 🛠️ Stack Technique

| Catégorie | Outil | Description |
| :--- | :--- | :--- |
| **Framework DL** | TensorFlow / Keras | 2.x - Construction et entraînement du modèle amélioré |
| **MLOps** | MLflow | Tracking avancé des expérimentations avec métriques de validation |
| **Langage** | Python | 3.9+ - Langage de développement principal |

---

## ⚙️ Structure du Projet

```text
tp2-improving-deep-models/
├── src/
│   └── train.py      # Script d'entraînement avec améliorations (L2, BN, Early Stopping)
├── models/           # Répertoire de sortie pour le modèle sauvegardé
├── report/           # Rapport du TP au format LaTeX (main.tex)
├── Dockerfile        # Fichier d'instructions pour construire le conteneur
├── requirements.txt  # Dépendances Python nécessaires au projet
└── README.md         # Ce fichier
```

---

## 🚀 Instructions d'Exécution

### Prérequis

* [Git](https://git-scm.com/)
* [Python 3.9+](https://www.python.org/downloads/)
* Environnement virtuel activé avec les dépendances installées

### 1. Configuration et Entraînement

**NOTE :** Assurez-vous que l'environnement virtuel est activé et que les dépendances sont installées (voir le `requirements.txt` à la racine du monorepo).

```bash
# Se placer dans le répertoire du TP
cd deep-learning-engineering-labs/tp2-improving-deep-models

# Entraîner le modèle amélioré et logguer l'expérimentation
python src/train.py
```

### 2. Suivi des Expérimentations (MLflow)

Après l'exécution, les résultats sont stockés dans le dossier `mlruns/`. Lancez l'interface utilisateur pour visualiser l'expérience :

```bash
# Remonter à la racine du monorepo (où se trouve le dossier mlruns/)
cd .. 

# Lancer le serveur MLflow
mlflow ui

# Accédez à l'interface via votre navigateur à http://127.0.0.1:5000
```

Dans l'interface MLflow, vous pouvez :
- Comparer les runs du TP1 et TP2
- Visualiser l'évolution des métriques de validation
- Analyser l'impact des différents hyperparamètres

### 3. Conteneurisation avec Docker

Le projet inclut un Dockerfile pour permettre l'exécution dans un environnement conteneurisé.

**Construction de l'image :**
```bash
# Se placer dans le répertoire du TP
cd deep-learning-engineering-labs/tp2-improving-deep-models

# Construire l'image Docker
docker build -t tp2-mnist-training:latest .
```

**Exécution du conteneur :**
```bash
# Exécuter le script d'entraînement dans le conteneur
docker run --rm tp2-mnist-training:latest

# Exécuter avec un volume pour persister les résultats MLflow
docker run --rm -v $(pwd)/mlruns:/app/mlruns tp2-mnist-training:latest

# Exécuter avec un volume pour sauvegarder les modèles
docker run --rm \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/models:/app/models \
  tp2-mnist-training:latest
```

**Avantages de la conteneurisation :**
- Reproductibilité : Environnement identique sur toutes les machines
- Isolation : Dépendances isolées du système hôte
- Portabilité : Exécutable sur n'importe quelle machine avec Docker
- CI/CD : Facilite l'intégration dans des pipelines d'automatisation

### 4. Comparaison avec le Modèle Baseline (TP1)

Pour comparer les performances :

1. **Via MLflow UI :**
   - Ouvrir l'interface MLflow
   - Sélectionner les expériences "TP1_MNIST_Deep_Learning_LifeCycle" et "TP2_Improving_Deep_Models"
   - Comparer les métriques `test_accuracy` et `test_loss`

2. **Via le code :**
   ```python
   import mlflow
   
   # Charger les runs
   client = mlflow.tracking.MlflowClient()
   tp1_runs = client.search_runs(experiment_ids=["TP1_experiment_id"])
   tp2_runs = client.search_runs(experiment_ids=["TP2_experiment_id"])
   
   # Comparer les métriques
   ```

---

## 📈 Analyse des Résultats

### Interprétation des Métriques

1. **Test Accuracy (97.62%)** : Précision finale sur le jeu de test, indicateur principal de performance
2. **Best Val Accuracy (97.82%)** : Meilleure précision atteinte sur le jeu de validation
3. **Gap Train-Val** : Différence entre train et validation (indicateur de surapprentissage)
   - Dans notre cas, le gap est faible, indiquant une bonne généralisation

### Impact des Techniques

1. **Régularisation L2** : 
   - Réduit le surapprentissage en pénalisant les poids élevés
   - Améliore la généralisation

2. **Batch Normalization** :
   - Stabilise l'entraînement
   - Permet une convergence plus rapide et stable

3. **Early Stopping** :
   - Évite le surapprentissage en arrêtant l'entraînement au bon moment
   - Économise le temps de calcul

4. **Dropout augmenté (0.2 → 0.3)** :
   - Réduit davantage le risque de surapprentissage
   - Force le modèle à être plus robuste

---

## 🔬 Expérimentations Futures

Pour aller plus loin, vous pouvez expérimenter avec :

1. **Hyperparamètres à ajuster :**
   - `l2_lambda` : Tester différentes valeurs (0.0001, 0.001, 0.01)
   - `dropout_rate` : Tester différentes valeurs (0.2, 0.3, 0.4, 0.5)
   - `early_stopping_patience` : Ajuster selon les besoins

2. **Architecture :**
   - Ajouter des couches supplémentaires
   - Tester différentes tailles de couches cachées
   - Expérimenter avec différentes fonctions d'activation

3. **Optimisation :**
   - Tester différents optimiseurs (RMSprop, SGD avec momentum)
   - Ajuster le learning rate
   - Implémenter un learning rate scheduler

---

## 📝 Notes Techniques

### Ordre des Couches

L'ordre recommandé pour les couches est :
1. **Dense** (avec régularisation L2)
2. **Batch Normalization** (si activé)
3. **Activation** (ReLU)
4. **Dropout**

Dans notre implémentation, Batch Normalization est placé après Dense et avant Dropout, ce qui est une pratique courante.

### Early Stopping

L'early stopping surveille `val_loss` et s'arrête si aucune amélioration n'est observée pendant `patience` epochs. Les meilleurs poids sont automatiquement restaurés grâce à `restore_best_weights=True`.

---

## 🎓 Conclusion

Le TP2 démontre l'importance des techniques de régularisation et d'optimisation pour améliorer les performances des modèles de deep learning. Les techniques implémentées (L2, Batch Normalization, Early Stopping) permettent d'obtenir un modèle plus robuste et généralisable.

**Points clés à retenir :**
- La régularisation est essentielle pour éviter le surapprentissage
- Batch Normalization stabilise et accélère l'entraînement
- Early Stopping optimise le temps d'entraînement tout en préservant les performances
- Le tracking MLflow permet de comparer et analyser les différentes expérimentations

---

## 📚 Références

- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Deep Learning Book - Regularization](https://www.deeplearningbook.org/contents/regularization.html)

