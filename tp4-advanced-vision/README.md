🩺 TP4 : Segmentation avancée (U-Net) et données 3D
==================================================

## 1. Contexte (continuation TP1→TP2→TP3→TP4)
- **TP1** : MLP sur MNIST, logging MLflow, conteneurisation.
- **TP2** : Régularisation (L2, Dropout, BatchNorm), EarlyStopping, tracking avancé.
- **TP3** : CNN sur CIFAR-10, blocs résiduels, vision 2D.
- **TP4** : Segmentation sémantique avec U-Net (médical) + introduction Conv3D (volumique). On réutilise les bonnes pratiques MLOps : structuration, Docker, tracking MLflow, métriques custom.

## 2. Objectifs pédagogiques
- Comprendre la sortie d’un modèle de segmentation (carte H×W×classes).
- Implémenter U-Net (encoder/decoder + skip concatenation).
- Utiliser des pertes et métriques adaptées : Dice, IoU, BCE+Dice.
- Tracer les expériences (MLflow) avec noms explicites d’architecture/optimiseur/perte.
- Introduire Conv3D pour données volumétriques et ses contraintes mémoire.

## 3. Structure du projet
```text
tp4-advanced-vision/
├── src/
│   └── unet_segmentation.py   # U-Net 2D, métriques Dice/IoU, MLflow, démo Conv3D
├── report/
│   └── main.tex               # Rapport théorique + lien repo
├── requirements.txt           # Dépendances (TF, numpy, mlflow…)
├── Dockerfile                 # Entraînement/expérimentation en conteneur
└── models/                    # (optionnel) sauvegardes de modèles
```

## 4. Installation
```bash
cd tp4-advanced-vision
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Script U-Net (src/unet_segmentation.py)
- **Prétraitement** : charge des données simulées si aucun dataset fourni (`--use-dummy-data` par défaut).
- **Modèle** : U-Net 2D avec conv_block (Conv-BN-ReLU), encoder (MaxPool), decoder (Conv2DTranspose), skip concatenation.
- **Pertes/Métriques** : Dice, IoU, BCE, BCE+Dice (passez `--loss bce_dice` pour les cas déséquilibrés).
- **MLflow** : `TP4_Segmentation_Unet` avec nom de run explicite (`unet_<loss>_<opt>_img<input>`), log des hyperparams, métriques finales, et sauvegarde du modèle.
- **Conv3D démo** : bloc simple avec logging MLflow (`--do-conv3d-demo`) pour illustrer la volumétrie.

### Exécution rapide (données simulées)
```bash
python src/unet_segmentation.py --epochs 3 --batch-size 8
```
Options utiles :
- `--loss {bce,dice,bce_dice}` (par défaut bce_dice)
- `--no-mlflow` pour désactiver le logging
- `--do-conv3d-demo` pour lancer l’exemple Conv3D + MLflow

### Brancher vos données
- Remplacez la fonction `load_data()` pour retourner `(x_train, y_train), (x_val, y_val), input_shape`.
- Les masques doivent être binaires (0/1) et alignés avec les images.

## 6. Docker
```bash
cd tp4-advanced-vision
docker build -t tp4-unet:latest .
docker run --rm tp4-unet:latest python src/unet_segmentation.py --epochs 3
```
- Montez un volume pour conserver modèles/logs : `-v $(pwd)/models:/app/models -v $(pwd)/mlruns:/app/mlruns`.

## 7. Résultats attendus (données simulées)
- Les métriques (Dice/IoU) sont surtout là pour vérifier le pipeline ; avec des données réelles, comparez les runs via MLflow et surveillez le déséquilibre foreground/background.

### Exemple de run (données simulées, CPU)
```
python src/unet_segmentation.py --epochs 3 --batch-size 8
```
Résultats observés (dummy data) :
- Train (fin epoch 3) : accuracy ≈ 0.91, Dice ≈ 0.78, IoU ≈ 0.64, loss ≈ 0.51
- Val (fin epoch 3) : accuracy ≈ 0.67, Dice ≈ 0.52, IoU ≈ 0.35, loss ≈ 1.15
- MLflow run : `unet_bce_dice_adam_img128` (expérience `TP4_Segmentation_Unet`)
⚠️ Warnings MLflow : `artifact_path` déprécié, absence de signature/input_example (peut être ajouté si vous fournissez un exemple d’entrée).

### Exemple de démo Conv3D
```
python src/unet_segmentation.py --do-conv3d-demo
```
Crée une expérience `TP4_Conv3D_Volumetric` avec un run `conv3d_baseline` (logging de la config modèle et métriques simulées).

## 8. Liens et rapport
- Rapport : `report/main.tex` (questions : sortie segmentation, rôle du decoder U-Net, différence des skips vs ResNet, pertes adaptées, métriques Dice/IoU, Conv3D et compromis mémoire).
- Repo : https://github.com/ThePerformer0/deep-learning-engineering-labs/tree/main/tp4-advanced-vision

## 9. Parallèle avec les TPs précédents
- Réutilisez L2/Dropout/BatchNorm (TP2) si surapprentissage.
- Réutilisez EarlyStopping/ReduceLROnPlateau pour stabiliser la convergence.
- Conservez la discipline MLflow (noms de runs explicites, log métriques custom).

## 10. Explications pédagogiques clés
- **Sortie segmentation** : carte H×W×C (ou H×W×1 en binaire), donc on optimise pixel par pixel, pas un seul label global.
- **U-Net vs ResNet** : U-Net concatène (skip) pour restaurer les détails perdus au pooling ; ResNet additionne pour faciliter le gradient dans un réseau très profond.
- **Pertes adaptées** : BCE seule pénalise peu les faux négatifs quand le foreground est minuscule ; BCE+Dice ou Dice améliorent le recouvrement sur les petites régions.
- **Métriques** : Dice (F1 segmentation) est plus indulgent que IoU ; IoU pénalise davantage les faux positifs et reste plus strict.
- **Conv3D** : noyaux 3D (kD×kH×kW) explorent aussi la profondeur (empilement de slices). Coût mémoire élevé → limiter filtres, taille de noyau, profondeur d’entrée, ou travailler par patchs/ROIs.
- **MLOps** : noms de runs explicites (`unet_<loss>_<opt>_img<input>`), log des hyperparams et métriques custom, sauvegarde du modèle. Montez un volume `mlruns/` dans Docker pour conserver l’historique.

