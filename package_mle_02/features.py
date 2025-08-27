from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from feast import FeatureStore

from package_mle_02.utils.helpers import *

import joblib

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "splitted_data.joblib",
    output_path: Path = PROCESSED_DATA_DIR / "train_test_data.joblib"
):
    
    # ---- Feature Engineering ----

    # ---- Loading splitted datasets ----
    logger.info("Loading datasets...")
    splitted_data = joblib.load(input_path)
    X_train = splitted_data["X_train"]
    logger.success("Loading datasets complete.")
    # -----------------------------------------

    # ---- Date Features ----
    logger.info("Date Features...")
    transforming_date_features(dataframe=X_train)
    logger.success("Date Features transformation complete.")
    # -----------------------------------------

    # ---- Nominal Encoding ----
    logger.info("Nominal Encoding...")
    X_train, _ = encoding_nominal_features(features_train=X_train, features_test=None, nominal_features=NOMINAL_FEATURES)
    logger.success("Nominal Encoding complete.")
    # -----------------------------------------

    # ---- Ordinal Encoding ----
    logger.info("Ordinal Encoding...")
    categories_list = [CUSTOM_ORDER[col] for col in ORDINAL_FEATURES]
    X_train, _ = encoding_ordinal_features(features_train=X_train, features_test=None, ordinal_features=ORDINAL_FEATURES, categories_list=categories_list)
    logger.success("Ordinal Encoding complete.")
    # -----------------------------------------

    # ---- Scaling ----
    logger.info("Scaling...")
    X_train, _ = scaling_numerical_features(method=SCALING_METHOD, features_train=X_train, features_test=None, features_to_scale=NUMERICAL_FEATURES+ORDINAL_FEATURES)
    logger.success("Scaling complete.")
    # -----------------------------------------

    # ---- Serving features with Feast ----
    logger.info("Serving features with Feast...")
    def writing_feature_table_aux(dataframe:pd.DataFrame, file_name:str):
        feature_table = dataframe.copy()
        if not feature_table.empty:
            feature_table[ID_FEATURE] = [str(uuid.uuid4()) for _ in range(feature_table.shape[0])]
            feature_table["event_timestamp"] = [datetime.now() for _ in range(feature_table.shape[0])]
            time.sleep(1)
            feature_table["created"] = [datetime.now() for _ in range(feature_table.shape[0])]
            feature_table.to_parquet(f"./feast_service/fs_mle_02/feature_repo/data/{file_name}.parquet", index=False)
            print(f"Feature Table in ./feast_service/fs_mle_02/feature_repo/data/{file_name}.parquet")
            return feature_table[ID_FEATURE].tolist()
        else:
            raise Exception("Feature table doesn't exist.")
    ids_to_retrieve_features = writing_feature_table_aux(dataframe=X_train, file_name="marketing_campaign_feature_table")
    logger.success("Serving features complete.")
    # -----------------------------------------

    # ---- Retrieving features with Feast ----
    logger.info("Retrieving features with Feast...")
    fs = FeatureStore("./feast_service/fs_mle_02/feature_repo/")
    entity_df = pd.DataFrame.from_dict({ID_FEATURE: ids_to_retrieve_features, "event_timestamp": [pd.Timestamp.now()] * len(ids_to_retrieve_features)})
    total_features = fs.get_historical_features(entity_df=entity_df, features=fs.get_feature_service("customer_marketing_campaign_feature_service")).to_df()
    """
    Getting a sample to reduce the training time (for academic purposes only)
    """
    X_train = total_features.sample(frac=0.05, random_state=SEED)
    logger.success("Retrieving features complete.")
    # -----------------------------------------

    # ---- Saving processed datasets ----
    logger.info("Saving artifacts...")
    processed_data = {
        "X_train": X_train.drop(columns=EXCLUDED_FEATURES)
    }
    joblib.dump(processed_data, output_path)
    logger.success("Saving artifacts complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
