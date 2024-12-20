import os
import pickle
import click
from sklearn.metrics import accuracy_score
import mlflow
from tensorflow.keras.models import load_model
import numpy as np

def load_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def load_h5_model(file_path):
    try:
        model = load_model(file_path)
        print(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

@click.command()
@click.option(
    "--data_path",
    default="./data",
    help="Location where the processed Sports data was saved"
)
def run_train(data_path: str):
    X_test, y_test = load_pickle(os.path.join(data_path, "test_data.pkl"))
    
    with mlflow.start_run():
        mlflow.set_tag("developer", "Aibek")
        mlflow.log_param("test-data-path", os.path.join(data_path, "test_data.pkl"))
        
        model = load_h5_model("models/my_model.h5")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Convert predictions to class labels
        y_pred_labels = y_pred.argmax(axis=1)
        
        # Ensure y_test is in label format
        if len(y_test.shape) > 1:  # If one-hot encoded
            y_test = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        acc_score_test = accuracy_score(y_test, y_pred_labels)
        mlflow.log_metric("accuracy_score_test", acc_score_test)
        print(f"Test accuracy: {acc_score_test}")
    

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("100-sport-cls")
    run_train()
