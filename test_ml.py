import pytest
# TODO: add necessary import
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, inference

def test_process_data():
    """
    Test the process_data function to ensure it correctly processes the data.
    """
    # Creating a sample dataframe
    data = pd.DataFrame({
        'workclass': ['Private', 'Self-emp-not-inc', 'Private'],
        'education': ['Bachelors', 'HS-grad', 'HS-grad'],
        'marital-status': ['Married-civ-spouse', 'Divorced', 'Divorced'],
        'occupation': ['Prof-specialty', 'Exec-managerial', 'Handlers-cleaners'],
        'relationship': ['Husband', 'Not-in-family', 'Not-in-family'],
        'race': ['White', 'Black', 'White'],
        'sex': ['Male', 'Female', 'Female'],
        'native-country': ['United-States', 'United-States', 'United-States'],
        'salary': ['>50K', '<=50K', '<=50K']
    })

    
    data.columns = [col.replace("-", "_") for col in data.columns]
    cat_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label='salary',
        training=True
    )

    # The shape of output
    assert X.shape[0] == 3
    expected_num_columns = X.shape[1] 
    assert X.shape[1] == expected_num_columns

    
    assert (y == lb.transform(data['salary'].values).ravel()).all()

def test_train_model():
    """
    Test the train_model function to ensure it correctly trains a model.
    """
    # Sample Dataframe
    X_train = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1, 2, 3, 4],
        'feature3': [5, 6, 7, 8]
    })
    y_train = pd.Series([0, 1, 0, 1])

    # Training the model
    model = train_model(X_train, y_train)

    
    assert isinstance(model, RandomForestClassifier)

    
    assert hasattr(model, "n_classes_")

    
    predictions = model.predict(X_train)
    assert len(predictions) == len(X_train)

def test_inference():
    """
    Test the inference function to ensure it correctly makes predictions.
    """
    # Sample dataframe
    X_train = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1, 2, 3, 4],
        'feature3': [5, 6, 7, 8]
    })
    y_train = pd.Series([0, 1, 0, 1])

    # Training the model
    model = train_model(X_train, y_train)

    # Sample dataframe
    X_test = pd.DataFrame({
        'feature1': [0.5, 0.6],
        'feature2': [5, 6],
        'feature3': [9, 10]
    })

    
    preds = inference(model, X_test)

    
    assert preds.shape[0] == X_test.shape[0]

    
    assert isinstance(preds, (pd.Series, np.ndarray))

if __name__ == "__main__":
    pytest.main()