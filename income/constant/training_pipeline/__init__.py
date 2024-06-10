import os
from income.constant.s3_bucket import TRAINING_BUCKET_NAME

SAVED_MODEL_DIR =os.path.join("saved_models")

# defining common constant variable for training pipeline
TARGET_COLUMN = "salary"
PIPELINE_NAME: str = "income"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "income.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
CORR_SCHEMA_FILE_PATH = os.path.join("config", "corr_schema.yaml")
TRAINED_FEATURES_PATH = os.path.join("config", "trained_features.yaml")

# defined accordingly.
SCHEMA_DROP_COLS = "drop_columns"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "income"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.33


"""  
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"



"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_SPEARMEN_THRESHOLD: float = .25




"""
Model Trainer ralated constant start with MODEL TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05




"""
Model Trainer ralated constant start with MODEL EVALUATION
"""
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02 # 2% improvement
MODEL_EVALUATION_REPORT_NAME= "report.yaml"


"""
Model Pusher related constant
"""

MODEL_PUSHER_DIR_NAME = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODEL_DIR

