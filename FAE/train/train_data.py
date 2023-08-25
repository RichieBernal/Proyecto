import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from preprocess.preprocess_data import (
    OneHotEncoder,
    FeatureSelector,
    OrderingFeatures
)

logger = logging.getLogger(__name__) # Indicamos que tome el nombre del modulo
logger.setLevel(logging.INFO) # Configuramos el nivel de logging

formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s') # Creamos el formato

file_handler = logging.FileHandler('FAE/logs/train.log') # Indicamos el nombre del archivo

file_handler.setFormatter(formatter) # Configuramos el formato

logger.addHandler(file_handler) # Agregamos el archivo

class FireDataPipeline:
    """
    A class representing the Titanic data processing and modeling pipeline.

    Attributes:
        NUMERICAL_VARS (list): A list of numerical variables in the dataset.
        CATEGORICAL_VARS (list): A list of categorical variables in the dataset.
        SEED_MODEL (int): A seed value for reproducibility.

    Methods:
        create_pipeline(): Create and return the Titanic data processing pipeline.
    """
    
    def __init__(self, seed_model, numerical_vars, categorical_vars, selected_features):
        self.SEED_MODEL = seed_model
        self.NUMERICAL_VARS = numerical_vars
        self.CATEGORICAL_VARS = categorical_vars
        self.SEED_MODEL = seed_model
        self.SELECTED_FEATURES = selected_features
        
        
    def create_pipeline(self):
        """
        Create and return the Titanic data processing pipeline.

        Returns:
            Pipeline: A scikit-learn pipeline for data processing and modeling.
        """
        self.PIPELINE = Pipeline(
            [
                                ('dummy_vars', OneHotEncoder(variables=self.CATEGORICAL_VARS)),
                                ('feature_selector', FeatureSelector(self.SELECTED_FEATURES)),
                                ('aligning_feats', OrderingFeatures()),
                                ('scaling', MinMaxScaler()),
                              ]
        )
        logger.info("Pipeline ok")
        return self.PIPELINE
    
    def fit_logistic_regression(self, X_train, y_train):
        """
        Fit a Logistic Regression model using the predefined data preprocessing pipeline.

        Parameters:
        - X_train (pandas.DataFrame or numpy.ndarray): The training input data.
        - y_train (pandas.Series or numpy.ndarray): The target values for training.

        Returns:
        - logistic_regression_model (LogisticRegression): The fitted Logistic Regression model.
        """
        logistic_regression = LogisticRegression(C=0.0005, class_weight='balanced', random_state=self.SEED_MODEL)
        pipeline = self.create_pipeline()
        pipeline.fit(X_train, y_train)
        logistic_regression.fit(pipeline.transform(X_train), y_train)
        logger.info("Regresión lógística aplicada al entrenamiento")
        return logistic_regression
    
    def transform_test_data(self, X_test):
        """
        Apply the data preprocessing pipeline on the test data.

        Parameters:
        - X_test (pandas.DataFrame or numpy.ndarray): The test input data.

        Returns:
        - transformed_data (pandas.DataFrame or numpy.ndarray): The preprocessed test data.
        """
        pipeline = self.create_pipeline()
        logger.info("Regresión lógística aplicada a la prueba")
        return pipeline.transform(X_test)