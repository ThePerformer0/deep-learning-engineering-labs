import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Évite les logs verbeux de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# --- 1. Chargement et prétraitement CIFAR-10 ---
def load_cifar10():
    """Charge CIFAR-10, normalise les pixels et applique le one-hot encoding."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    num_classes = 10
    input_shape = x_train.shape[1:]  # (32, 32, 3)

    # Normalisation en [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # One-hot encoding des labels
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    print(f"Input data shape: {input_shape}")
    print(f"Train labels shape: {y_train.shape}, Test labels shape: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


# --- 2. CNN basique ---
def build_basic_cnn(input_shape, num_classes):
    """Architecture CNN classique (2 blocs conv/pooling) pour CIFAR-10."""
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=input_shape
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


# --- 3. Bloc résiduel simplifié ---
def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    """Bloc résiduel : conv -> conv + skip connection (option stride)."""
    y = keras.layers.Conv2D(
        filters, kernel_size, strides=stride, padding="same", activation="relu"
    )(x)
    y = keras.layers.Conv2D(filters, kernel_size, padding="same")(y)

    if stride > 1:
        # Adapter la dimension spatiale du skip si stride > 1
        x = keras.layers.Conv2D(filters, (1, 1), strides=stride, padding="same")(x)

    out = keras.layers.Add()([x, y])
    out = keras.layers.Activation("relu")(out)
    return out


def build_small_resnet(input_shape, num_classes):
    """Petit réseau avec 3 blocs résiduels pour illustration."""
    inputs = keras.Input(shape=input_shape)
    x = residual_block(inputs, 32)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def train_and_eval(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=10):
    """Compile, entraîne et évalue un modèle."""
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    return history, test_acc, test_loss


def main(use_resnet=False):
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_cifar10()

    if use_resnet:
        print("Training small ResNet model...")
        model = build_small_resnet(input_shape, num_classes)
    else:
        print("Training basic CNN model...")
        model = build_basic_cnn(input_shape, num_classes)

    model.summary()
    train_and_eval(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=10)


if __name__ == "__main__":
    # Mettre use_resnet=True pour tester la version résiduelle
    main(use_resnet=False)

