from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np
import joblib
from package_mle_02.utils.helpers import *

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "new_marketing_campaign.csv",
    output_path: Path = PROCESSED_DATA_DIR / "splitted_data.joblib"
):
    
    # ---- Loading and Cleaning ----
    logger.info("Loading dataset...")
    df_marketing_campaign_raw = pd.read_csv(input_path)
    logger.success("Loading dataset complete.")

    logger.info("Cleaning dataset...")
    df_marketing_campaign = df_marketing_campaign_raw.copy()
    df_marketing_campaign.drop(DROP_FEATURES, axis=1, inplace=True)
    df_marketing_campaign = dropping_invalid_year_birth_rows(dataframe=df_marketing_campaign)
    df_marketing_campaign = stripping_object_features(dataframe=df_marketing_campaign)
    df_marketing_campaign = casting_numerical_features(dataframe=df_marketing_campaign)
    logger.success("Cleaning dataset complete.")
    # -----------------------------------------

    # ---- Data Splitting ----
    logger.info("Splitting dataset...")
    X_train, _, _, _ = split_dataset(target_feature=TARGET, dataframe=df_marketing_campaign,
                                     test_percentage=TEST_PERCENTAGE, seed=SEED, stratify_feature=STRATIFY_FEATURE)
    logger.success("Splitting dataset complete.")
    # -----------------------------------------

    # ---- Saving datasets ----
    logger.info("Splitting dataset...")
    splitted_data = {
        "X_train": X_train
    }
    joblib.dump(splitted_data, output_path)
    logger.success("Saving datasets complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
