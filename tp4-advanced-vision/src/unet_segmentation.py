import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import mlflow
import mlflow.keras


# Configuration silencieuse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ---------------------------
# Métriques et pertes
# ---------------------------
def dice_coeff(y_true, y_pred, smooth=1.0):
    # Utilise reshape pour éviter l'absence de K.flatten dans certaines versions
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# ---------------------------
# Blocs de modèle
# ---------------------------
def conv_block(x, filters):
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def build_unet(input_shape=(128, 128, 1), num_classes=1, base_filters=32):
    inputs = keras.Input(input_shape)

    # Encodeur
    c1 = conv_block(inputs, base_filters)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, base_filters * 4)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, base_filters * 8)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    b = conv_block(p4, base_filters * 16)

    # Decodeur
    u1 = keras.layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(b)
    u1 = keras.layers.Concatenate()([u1, c4])
    d1 = conv_block(u1, base_filters * 8)

    u2 = keras.layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(d1)
    u2 = keras.layers.Concatenate()([u2, c3])
    d2 = conv_block(u2, base_filters * 4)

    u3 = keras.layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(d2)
    u3 = keras.layers.Concatenate()([u3, c2])
    d3 = conv_block(u3, base_filters * 2)

    u4 = keras.layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(d3)
    u4 = keras.layers.Concatenate()([u4, c1])
    d4 = conv_block(u4, base_filters)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = keras.layers.Conv2D(num_classes, (1, 1), activation=activation)(d4)
    return keras.Model(inputs, outputs, name="unet_2d")


# ---------------------------
# Chargement / données simulées
# ---------------------------
def load_dummy_data(samples=32, input_shape=(128, 128, 1)):
    """Génère des données simulées pour tester le pipeline."""
    x = np.random.rand(samples, *input_shape).astype("float32")
    # Masques binaires grossiers
    y = (x > 0.5).astype("float32")
    # Split train/val 75/25
    split = int(samples * 0.75)
    return (x[:split], y[:split]), (x[split:], y[split:])


# ---------------------------
# Compilation / entraînement
# ---------------------------
def compile_model(model, loss_name="bce_dice", lr=1e-3):
    if loss_name == "dice":
        loss_fn = dice_loss
    elif loss_name == "bce":
        loss_fn = "binary_crossentropy"
    else:  # bce_dice
        def bce_dice(y_true, y_pred):
            return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
        loss_fn = bce_dice

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[dice_coeff, iou_metric, "accuracy"],
    )
    return model


def train_unet(args):
    if args.use_dummy_data:
        (x_train, y_train), (x_val, y_val) = load_dummy_data(samples=32, input_shape=args.input_shape)
    else:
        raise NotImplementedError("Remplacez load_dummy_data par votre chargement de dataset.")

    model = build_unet(input_shape=args.input_shape, num_classes=1, base_filters=args.base_filters)
    compile_model(model, loss_name=args.loss, lr=args.learning_rate)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    val_metrics = {
        "val_loss": float(history.history["val_loss"][-1]),
        "val_dice": float(history.history["dice_coeff"][-1]),
        "val_iou": float(history.history["iou_metric"][-1]),
    }
    return model, val_metrics


# ---------------------------
# Démo Conv3D + MLflow
# ---------------------------
def simple_conv3d_block(input_shape=(32, 32, 32, 1)):
    inputs = keras.Input(input_shape)
    x = keras.layers.Conv3D(16, (3, 3, 3), activation="relu", padding="same")(inputs)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)
    x = keras.layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="conv3d_block_demo")


def run_conv3d_demo(log_mlflow=True):
    mlflow.set_experiment("TP4_Conv3D_Volumetric")
    with mlflow.start_run(run_name="conv3d_baseline"):
        model_3d = simple_conv3d_block()
        model_config = model_3d.to_json()
        mlflow.log_dict({"model_config": model_config}, "artifacts/model_architecture.json")
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("filters_start", 16)
        # Simulation d'un résultat
        mlflow.log_metric("val_loss", 0.5)
        mlflow.log_metric("val_acc", 0.8)
        print("MLflow tracking complete for Conv3D demo.")


# ---------------------------
# MLflow helper
# ---------------------------
def log_with_mlflow(model, args, val_metrics):
    mlflow.set_experiment("TP4_Segmentation_Unet")
    run_name = f"unet_{args.loss}_{args.optimizer}_img{args.input_shape[0]}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "loss": args.loss,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "input_shape": args.input_shape,
                "base_filters": args.base_filters,
            }
        )
        for k, v in val_metrics.items():
            mlflow.log_metric(k, v)
        mlflow.keras.log_model(model, "model")
        print(f"MLflow run logged: {run_name}")


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="U-Net segmentation (TP4)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loss", type=str, default="bce_dice", choices=["bce", "dice", "bce_dice"])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-shape", type=int, nargs=3, default=[128, 128, 1])
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--no-mlflow", action="store_true", help="Désactive le logging MLflow")
    parser.add_argument("--use-dummy-data", action="store_true", help="Utiliser des données simulées", default=True)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--do-conv3d-demo", action="store_true", help="Lancer la démo Conv3D + MLflow")
    return parser.parse_args()


def main():
    args = parse_args()

    # Forcer bool pour input_shape
    args.input_shape = tuple(args.input_shape)

    if args.do_conv3d_demo:
        run_conv3d_demo(log_mlflow=not args.no_mlflow)
        return

    model, val_metrics = train_unet(args)
    if not args.no_mlflow:
        log_with_mlflow(model, args, val_metrics)


if __name__ == "__main__":
    main()

