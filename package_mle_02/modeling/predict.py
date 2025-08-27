from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from package_mle_02.utils.helpers import *

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_test_data.joblib",
    fitted_models_path: Path = MODELS_DIR,
    base_model_labels_path: Path = PROCESSED_DATA_DIR / "base_model_labels.csv",
    model_01_labels_path: Path = PROCESSED_DATA_DIR / "model_01_labels.csv",
    model_02_labels_path: Path = PROCESSED_DATA_DIR / "model_02_labels.csv"
):

    # ---- Loading test datasets----
    logger.info("Loading datasets to predict...")
    train_test_data = joblib.load(features_path)
    X_train = train_test_data["X_train"]
    """
    Getting a sample to simulate predictions
    """
    X_sample = X_train.sample(frac=0.005, random_state=SEED)
    logger.success("Loading datasets to predict complete.")
    # -----------------------------------------

    # ---- Loading fitted models ----
    logger.info("Loading fitted models...")
    base_model = joblib.load(fitted_models_path/f"{BASE_MODEL_NAME}_model.joblib")
    model_01 = joblib.load(fitted_models_path/f"{MODEL_01_NAME}_model.joblib")
    model_02 = joblib.load(fitted_models_path/f"{MODEL_02_NAME}_model.joblib")
    logger.success("Loading fitted models complete.")
    # -----------------------------------------

    # ---- Predicting ----
    logger.info("Predicting...")
    mlflow.autolog(log_models=True,)
    base_model_labels = base_model.predict(X_sample)
    model_01_labels = model_01.predict(X_sample)
    model_02_labels = model_02.predict(X_sample)
    logger.success("Predictions complete.")
    # -----------------------------------------

    # ---- Saving labels ----
    logger.info("Saving labels...")
    pd.DataFrame(base_model_labels, columns=["cluster"]).to_csv(base_model_labels_path, index=False)
    pd.DataFrame(model_01_labels, columns=["cluster"]).to_csv(model_01_labels_path, index=False)
    pd.DataFrame(model_02_labels, columns=["cluster"]).to_csv(model_02_labels_path, index=False)
    logger.success("Saving labels complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
