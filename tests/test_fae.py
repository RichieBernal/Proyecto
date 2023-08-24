#!/usr/bin/env python

"""Tests for `FAE` package."""
import os
import pytest
import pandas as pd

from sklearn.pipeline import Pipeline

from FAE.load.load_data import DataRetriever
from FAE.preprocess.preprocess_data import OneHotEncoder

def test_one_hot_encoder_transform():
    """
    Test the `transform` method of the OneHotEncoder transformer.

    This test checks if the transformer to convert categorical variables into a numerical representation creating binary columns for each category in the categorical variable.

    The test case uses a sample DataFrame with cat values and a custom transformer instance.

    It checks if the transformer successfully adds columns for the specified values.

    Note: Make sure to replace 'your_module' with the actual module name where the MissingIndicator class is defined.
    """
    # Sample DataFrame with missing values
    data = {
        'FUEL': ['gasoline', 'kerosene', 'thinner']
    }
    df = pd.DataFrame(data)

    # Instantiate the custom transformer with specified variables
    dummy_vars = OneHotEncoder(variables=['FUEL'])

    # Transform the DataFrame using the custom transformer
    df_transformed = dummy_vars.transform(df)

    # Check if the transformed DataFrame has the expected additional columns
    expected_columns = ['kerosene', 'thinner']
    assert all(col in df_transformed.columns for col in expected_columns), \
        f"The transformed DataFrame should have the following additional columns: {expected_columns}"

    # Check if the original DataFrame is not modified
    assert 'thinner' not in df.columns, "The original DataFrame should not be modified."

def does_csv_file_exist(file_path):
    """
    Check if a CSV file exists at the specified path.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)

def test_csv_file_existence():
    """
    Test case to check if the CSV file exists.
    """
    # Provide the path to the CSV file that needs to be tested
    csv_file_path = "FAE/data/data_fire.csv"
    
    DATASETS_DIR = './data/'
    
    URL = 'C:/Users/rbernal/Documents/GitHub/Proyecto/FAE/data/data_fire.csv'
    data_retriever = DataRetriever(URL, DATASETS_DIR)
    data_retriever.retrieve_data()

    # Call the function to check if the CSV file exists
    file_exists = does_csv_file_exist(csv_file_path)

    # Use Pytest's assert statement to check if the file exists
    assert file_exists == True, f"The CSV file at '{csv_file_path}' does not exist."

def test_model_existence():
    """
    Test to validate the existence of a .pkl model file.

    This test function checks whether the specified .pkl model file exists
    in the specified directory.

    Raises:
        AssertionError: If the model file doesn't exist.

    Usage:
        Run this test using the pytest command:
        pytest test_model_existence.py
    """
    model_filename = "logistic_regression_output.pkl"
    MODEL_DIRECTORY = "FAE/models"
    model_path = os.path.join(MODEL_DIRECTORY, model_filename)
    print(model_path)
    assert os.path.exists(model_path), f"Model file '{model_filename}' does not exist."

if __name__ == "__main__":
    # Run the test function using Pytest
    pytest.main([__file__])

