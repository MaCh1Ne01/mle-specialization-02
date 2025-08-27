from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from package_mle_02.utils.helpers import *

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_test_data.joblib",
    fitted_models_path: Path = MODELS_DIR / "fitted_models.joblib"
):
    
    # ---- Loading processed train datasets ----
    logger.info("Loading processed train datasets...")
    train_test_data = joblib.load(features_path)
    X_train = train_test_data["X_train"]
    logger.success("Loading processed train datasets complete.")
    # -----------------------------------------

    # ---- DagsHub Integration ----
    logger.info("Connecting with DagsHub...")
    dagshub.init(repo_owner=REPOSITORY_OWNER, repo_name=REPOSITORY_NAME, mlflow=True)
    """
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'MaCh1Ne01'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '[GENERATED_TOKEN]'
    """
    logger.success("Connection with DagsHub succesfully.")
    # -----------------------------------------

    # ---- MLflow Integration ----
    logger.info("Connecting with MLflow...")
    mlflow.set_tracking_uri(MLFLOW_DAGSHUB_URL)
    setting_experiment()
    mlflow.autolog(log_models=True,)
    logger.success("Connection with MLflow succesfully.")
    # -----------------------------------------

    # ---- Models ----
    logger.info("Initializing models...")
    number_of_clusters = 5
    base_model = GaussianMixture(n_components=number_of_clusters, random_state=SEED)
    model_01 = KMeans(n_clusters=number_of_clusters, random_state=SEED)
    model_02 = OPTICSKMeansEnsemble(n_clusters=number_of_clusters, random_state=SEED)
    logger.success("Initializing models complete.")
    # -----------------------------------------

    # ---- Fitting and Saving Models ----
    logger.info("Fitting and Saving models...")
    def executing_and_saving_clustering_model_aux(model:any, model_name:str, X:pd.DataFrame, label_data:str):   
        with mlflow.start_run(run_name=f"{model_name} Model Run") as run:
            labels = model.fit_predict(X)
            mlflow.log_params(model.get_params())
            joblib.dump(model, f"./models/{model_name}_model.joblib")
            # Filtering only clusterized points
            valid_mask = labels != -1
            X_valid = X[valid_mask]
            labels_valid = labels[valid_mask]
            if len(np.unique(labels_valid)) > 1:
                silhouette = silhouette_score(X_valid, labels_valid)
                calinski_harabasz = calinski_harabasz_score(X_valid, labels_valid)
                davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
                mlflow.log_metrics(
                    {
                        "Silhouette Score": silhouette,
                        "Calinski-Harabasz Score": calinski_harabasz,
                        "Davies-Bouldin Score:": davies_bouldin
                    }   
                )
            else:
                raise ValueError("Only 1 cluster exists.")
        print(f"**********{model_name} Metrics ({label_data}):**********")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        return silhouette, calinski_harabasz, davies_bouldin, labels
    executing_and_saving_clustering_model_aux(model=base_model, model_name=BASE_MODEL_NAME, X=X_train, label_data=TRAINING_DATA_LABEL)
    executing_and_saving_clustering_model_aux(model=model_01, model_name=MODEL_01_NAME, X=X_train, label_data=TRAINING_DATA_LABEL)
    executing_and_saving_clustering_model_aux(model=model_02, model_name=MODEL_02_NAME, X=X_train, label_data=TRAINING_DATA_LABEL)
    logger.success("Fitting and Saving models complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
