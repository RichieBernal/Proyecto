"""Main module."""
from load.load_data import DataRetriever
from train.train_data import FireDataPipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score

     
DATASETS_DIR = './data/'
URL = 'C:/Users/rbernal/Documents/GitHub/Proyecto/FAE/data/data_fire.csv'
RETRIEVED_DATA = 'data_fire.csv'


SEED_SPLIT = 404
TRAIN_DATA_FILE = DATASETS_DIR + 'train.csv'
TEST_DATA_FILE  = DATASETS_DIR + 'test.csv'

TARGET  = 'STATUS'
FEATURES = ['SIZE','FUEL','DISTANCE','DESIBEL','AIRFLOW','FREQUENCY']
CATEGORICAL_VARS = ['FUEL']
NUMERICAL_VARS = ['SIZE','DISTANCE','DESIBEL','AIRFLOW','FREQUENCY']

SEED_MODEL = 404

SELECTED_FEATURES = ['SIZE',
                     'FUEL', 
                     #'FUEL_gasoline',
                     'FUEL_lpg', 
                     'FUEL_kerosene',
                     'FUEL_thinner',
                     'DISTANCE','DESIBEL','AIRFLOW','FREQUENCY']

TRAINED_MODEL_DIR = './models/'
PIPELINE_NAME = 'logistic_regression'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'

if __name__ == "__main__":
    
#    print(os.getcwd())
    os.chdir('./FAE')
    # Retrieve data
    data_retriever = DataRetriever(URL, DATASETS_DIR)
    result = data_retriever.retrieve_data()
#    print(result)
    
    # Instantiate the FireDataPipeline class
    Fire_data_pipeline = FireDataPipeline(seed_model=SEED_MODEL,
                                                numerical_vars=NUMERICAL_VARS, 
                                                categorical_vars=CATEGORICAL_VARS,
                                                selected_features=SELECTED_FEATURES)
    
    # Read data
    df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
                                                        df.drop(TARGET, axis=1),
                                                        df[TARGET],
                                                        test_size=0.2,
                                                        random_state=404
                                                   )
    
    
    logistic_regression_model = Fire_data_pipeline.fit_logistic_regression(X_train, y_train)
    
    X_test = Fire_data_pipeline.PIPELINE.fit_transform(X_test)
    y_pred = logistic_regression_model.predict(X_test)
    
    class_pred = logistic_regression_model.predict(X_test)
    proba_pred = logistic_regression_model.predict_proba(X_test)[:,1]
    print(f'test roc-auc : {roc_auc_score(y_test, proba_pred)}')
    print(f'test accuracy: {accuracy_score(y_test, class_pred)}')
    
    # # Save the model using joblib
    save_path = TRAINED_MODEL_DIR + PIPELINE_SAVE_FILE
    joblib.dump(logistic_regression_model, save_path)
    print(f"Model saved in {save_path}")
