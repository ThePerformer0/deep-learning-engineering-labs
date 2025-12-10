🧪 TP3 : CNN & Vision (CIFAR-10)
================================

## 1. Contexte et continuité (TP1 → TP2 → TP3)
- **TP1 (MNIST, MLP + Dropout)** : bases du DL, logging MLflow, conteneurisation.
- **TP2 (MNIST amélioré)** : régularisation L2, BatchNorm, EarlyStopping, suivi avancé.
- **TP3 (CIFAR-10, CNN)** : on passe aux images couleur 32×32×3. Les convolutions exploitent la structure spatiale et les blocs résiduels facilitent l’entraînement de réseaux plus profonds. On réutilise les réflexes d’industrialisation vus en TP1/TP2 (structure projet, Docker, métriques).

## 2. Notions clés et utilité
- **Convolution + stride/padding** : extraire des motifs locaux (bords, textures) avec partage de poids → bien plus efficace que du Dense sur des images.
- **Pooling (Max/Avg)** : réduire la dimension et gagner en robustesse aux translations.
- **Flatten / GlobalAveragePooling** : passer des cartes de features aux couches denses pour la décision finale.
- **Blocs résiduels (ResNet)** : skip connection pour limiter le vanishing gradient et permettre des réseaux plus profonds/stables.
- **CIFAR-10** : plus complexe que MNIST (couleur, fonds variés) → montre l’intérêt des convolutions.
- **Parallèle TP2** : on peut ajouter L2/Dropout/BatchNorm sur conv et denses comme dans le TP2 si besoin de régularisation.

## 3. Structure du projet
```text
tp3-cnn-vision/
├── src/
│   └── cnn_classification.py   # Prétraitement CIFAR-10, CNN basique, mini-ResNet
├── report/
│   └── main.tex                # Rapport TP3 (questions théoriques + résultats)
├── requirements.txt            # Dépendances locales (TF, numpy, matplotlib, pillow)
├── Dockerfile                  # Image pour entraîner le CNN
└── models/                     # (optionnel) sauvegardes de modèles
```

## 4. Installation rapide
```bash
cd tp3-cnn-vision
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Exécution (CPU par défaut)
```bash
cd tp3-cnn-vision
python src/cnn_classification.py          # CNN basique (2 conv + 2 pool + denses)
```
- Pour tester le mini-ResNet illustratif : ouvrir `cnn_classification.py` et passer `use_resnet=True` dans `main()`.
- Entraînement par défaut : 10 époques, batch_size=64, validation_split=0.1.
- Téléchargement CIFAR-10 : ~170 Mo au premier lancement.

## 6. Ce que fait le script
1) **Chargement/Prétraitement** : normalisation [0,1], one-hot des labels, affichage des shapes.  
2) **Modèle basique** : Conv(32, 3×3, same) → MaxPool(2×2) → Conv(64, 3×3, same) → MaxPool → Flatten → Dense(512) → Dense(10, softmax).  
3) **Mini-ResNet (option)** : 3 blocs résiduels (32) ; (64, stride 2) ; (64) + GlobalAveragePooling + Dense.  
4) **Entraînement** : Adam + categorical_crossentropy, suivi accuracy/val_accuracy.  
5) **Évaluation** : affiche test accuracy / test loss.

## 7. Parallèle avec TP1/TP2 (réutiliser les bonnes pratiques)
- **Régularisation** : ajouter L2/Dropout/BatchNorm si surapprentissage (cf. TP2).
- **EarlyStopping + logging** : reprendre le callback d’early stopping et le tracking MLflow du TP2 si besoin de suivi d’expériences.
- **Conteneurisation** : même logique que TP1/TP2 pour reproductibilité.

## 8. Docker
```bash
cd tp3-cnn-vision
docker build -t tp3-cnn:latest .
docker run --rm tp3-cnn:latest
```
- Monter un volume pour conserver modèles/logs : `-v $(pwd)/models:/app/models`.

## 9. Résultats indicatifs (CPU)
- CNN basique 10 époques : précision test typique ~70 % (peut varier selon matériel/seed).
- Le mini-ResNet peut apporter une stabilité supplémentaire si vous ajoutez BatchNorm/L2.

## 10. Rapport LaTeX
- `report/main.tex` : réponses théoriques (convolution/pooling, skip connections, segmentation U-Net, détection bbox, style transfer avec VGG16).
- Compilation : `cd report && pdflatex main.tex`.

## 11. Pistes d’approfondissement
- Ajouter **BatchNorm** après les conv, **Dropout** après denses, **L2** sur kernels.
- Brancher **EarlyStopping** et un logger (MLflow / TensorBoard).
- Tester un **scheduler de learning rate** ou un optimiseur différent (SGD momentum).
- Sauvegarder le modèle dans `models/` (`model.save(...)`).

## 12. Rappels pratiques
- GPU (CUDA) recommandé pour accélérer l’entraînement CIFAR-10.
- Vérifier l’espace disque (download ~170 Mo).
- Sur CPU, réduire `epochs` si besoin pour des tests rapides.

