from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Constants
TARGET = None
CUTOFF_DATE = "2023-09-15"
TEST_PERCENTAGE = 0
SEED = 2
DROP_FEATURES = ["ID","Z_CostContact","Z_Revenue"]
SCALING_METHOD = "Robust"
REPOSITORY_OWNER = "MaCh1Ne01"
REPOSITORY_NAME = "mle-specialization-02"
MLFLOW_DAGSHUB_URL = "https://dagshub.com/MaCh1Ne01/mle-specialization-02.mlflow"
EXPERIMENT_NAME = "Customer Segmentation - MLE 02 Project"
OBJECT_TO_NUMERICAL_FEATURES = ["Year_Birth","Teenhome"]
INITIAL_OBJECT_FEATURES = ["Year_Birth","Education","Marital_Status","Teenhome","Dt_Customer"]
INITIAL_NUMERICAL_FEATURES = ["Year_Birth","Income","Kidhome","Teenhome","Recency","MntWines","MntFruits","MntMeatProducts",\
                      "MntFishProducts","MntSweetProducts","MntGoldProds","NumDealsPurchases","NumWebPurchases","NumCatalogPurchases",\
                      "NumStorePurchases","NumWebVisitsMonth","AcceptedCmp2","Response"]
NOMINAL_FEATURES = ["Marital_Status"]
ORDINAL_FEATURES = ["Education"]
NUMERICAL_FEATURES = ["Customer_Age","Income","Kidhome","Teenhome","Customer_Tenure","Recency","MntWines","MntFruits","MntMeatProducts",\
                      "MntFishProducts","MntSweetProducts","MntGoldProds","NumDealsPurchases","NumWebPurchases","NumCatalogPurchases",\
                      "NumStorePurchases","NumWebVisitsMonth","AcceptedCmp2","Response"]
DATE_FEATURES_TRANSFORM = [("Year_Birth","years"),("Dt_Customer","days")]
DATE_FEATURES_RENAMED = {"Year_Birth": "Customer_Age", "Dt_Customer": "Customer_Tenure"}
ID_FEATURE = "customer_marketing_campaign_id"
EXCLUDED_FEATURES = [ID_FEATURE] + ["event_timestamp","created"]
STRATIFY_FEATURE = None
CUSTOM_ORDER = {
    "Education": ["Basic","2n Cycle","Graduation","Master","PhD"]
}
FEAST_REPOSITORY_PATH = "../feast_service/fs_mle_02/feature_repo/"
BASE_MODEL_NAME = "Gaussian Mixture"
MODEL_01_NAME = "K-Means"
MODEL_02_NAME = "OPTICS+K-Means Ensemble"
TRAINING_DATA_LABEL = "Training Data"
TESTING_DATA_LABEL = None

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
