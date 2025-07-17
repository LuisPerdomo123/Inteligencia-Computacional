import os
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# Si el script se ejecuta en Google Colab, se monta Google Drive de forma
# automática para acceder al conjunto de datos.
try:  # pragma: no cover - instrucción específica para Colab
    from google.colab import drive

    drive.mount("/content/drive")
except Exception:
    # En entornos fuera de Colab simplemente continuamos.
    pass

# Ruta del conjunto de datos (puede sobrescribirse con la variable de entorno
# DATASET_PATH). Por defecto apunta a Google Drive.
DATASET_PATH = os.getenv(
    "DATASET_PATH", "/content/drive/My Drive/DataSet/IEEE13Polaridad_V7/"
)
MODEL_OUTPUT_PATH = os.getenv("MODEL_OUTPUT_PATH", "trained_model.h5")
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 25
FINE_TUNE_EPOCHS = 10


def create_dataframe_from_images(dataset_path: str) -> pd.DataFrame:
    """Create dataframe with image paths and labels."""
    image_generator = ImageDataGenerator(rescale=1.0 / 255)
    data_flow = image_generator.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    class_names = list(data_flow.class_indices.keys())
    df = pd.DataFrame(
        {
            "image_path": data_flow.filepaths,
            "label": [class_names[idx] for idx in data_flow.classes],
        }
    )
    return df


def visualize_dataset_per_class(df: pd.DataFrame, num_samples: int = 5) -> None:
    """Display a few samples per class."""
    classes = df["label"].unique()
    for class_name in classes:
        class_df = df[df["label"] == class_name].sample(num_samples)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Class: {class_name}", fontsize=16)
        for i, (_, row) in enumerate(class_df.iterrows(), 1):
            image_path = row["image_path"]
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.subplot(1, num_samples, i)
                plt.imshow(image)
                plt.axis("off")
        plt.show()


def compute_class_weights(generator) -> dict:
    """Compute class weights for imbalanced datasets."""
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(generator.classes),
        y=generator.classes,
    )
    return {i: w for i, w in enumerate(class_weights)}


def build_transfer_model(trainable: bool = False) -> models.Model:
    """Create the transfer learning model."""
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = trainable
    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def fine_tune_model(model: models.Model, base_model_layers: int) -> None:
    """Unfreeze top layers of the base model for fine tuning."""
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-base_model_layers]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])


def plot_roc_pr_curves(
    y_true: np.ndarray, y_pred_prob: np.ndarray, class_indices: dict
) -> None:
    """Plot ROC and PR curves for each class."""
    import matplotlib.pyplot as plt

    # Use the provided dictionary instead of relying on a global variable.
    for i, class_name in enumerate(class_indices.keys()):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC {class_name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "--")
        plt.title(f"ROC Curve — Class: {class_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_pred_prob[:, i])
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, label=f"PR {class_name} (AUC = {pr_auc:.2f})")
        plt.title(f"Precision-Recall Curve — Class: {class_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.show()


if __name__ == "__main__":
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"No se encontró la ruta de datos: {DATASET_PATH}")

    df_images = create_dataframe_from_images(DATASET_PATH)
    visualize_dataset_per_class(df_images)

    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    model = build_transfer_model(trainable=False)
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_OUTPUT_PATH, monitor="val_loss", save_best_only=True)

    class_weights = compute_class_weights(train_gen)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weights,
    )

    # Fine tuning: unfreeze top 50 layers and train with smaller learning rate
    fine_tune_model(model, base_model_layers=50)
    history_ft = model.fit(
        train_gen,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weights,
    )

    # Evaluation
    y_true = val_gen.classes
    y_pred_prob = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))
    print("Confusion Matrix:", confusion_matrix(y_true, y_pred))

    plot_roc_pr_curves(y_true, y_pred_prob, val_gen.class_indices)

    for i, class_name in enumerate(val_gen.class_indices.keys()):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        print(f"AUC for {class_name}: {roc_auc:.2f}")

    # Serialize model
    model.save(MODEL_OUTPUT_PATH)
