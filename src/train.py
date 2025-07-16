# train.py
# Script to train a logistic regression model on diabetes data
# It uses scikit-learn for model training and MLflow for tracking experiments

# ------------------ IMPORT REQUIRED LIBRARIES ------------------

import argparse  # For parsing command-line arguments
import glob      # For file pattern matching (e.g., *.csv)
import os        # For file path operations

import pandas as pd                 # For data handling
import numpy as np                  # For numerical operations
from sklearn.linear_model import LogisticRegression        # Our model
from sklearn.model_selection import train_test_split       # To split data
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve  # To evaluate model
import matplotlib.pyplot as plt     # To plot ROC curve

import mlflow                      # For experiment tracking
import mlflow.sklearn              # For auto-logging sklearn models

# ------------------ ENABLE MLFLOW AUTOLOGGING ------------------

# Automatically logs model parameters, metrics, and artifacts (like plots)
mlflow.sklearn.autolog()

# ------------------ MAIN FUNCTION ------------------

def main(args):
    """
    Main function that handles:
    - Data loading
    - Splitting into train/test sets
    - Training the model
    - Evaluating and saving the model
    """
    df = get_csvs_df(args.training_data)                 # Load all CSVs into one DataFrame
    X_train, X_test, y_train, y_test = split_data(df)    # Split features and target
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)  # Train and evaluate

# ------------------ READ CSV FILES ------------------

def get_csvs_df(path):
    """
    Reads all CSV files from the specified folder and concatenates them.

    Args:
        path (str): Folder path containing CSVs.

    Returns:
        pd.DataFrame: Combined data from all CSVs.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(f"{path}/*.csv")

    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")

    # Read all CSVs and merge them into one DataFrame
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

# ------------------ SPLIT DATA INTO TRAIN/TEST ------------------

def split_data(df):
    """
    Splits the dataframe into features (X) and target (y),
    then splits those into training and testing sets.

    Args:
        df (pd.DataFrame): DataFrame with all data

    Returns:
        X_train, X_test, y_train, y_test: Numpy arrays for training and testing
    """
    # Define feature columns
    X = df[[
        'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
        'TricepsThickness', 'SerumInsulin', 'BMI',
        'DiabetesPedigree', 'Age'
    ]].values

    # Target column
    y = df['Diabetic'].values

    # Split data into 70% train and 30% test
    return train_test_split(X, y, test_size=0.3, random_state=0)

# ------------------ TRAIN AND EVALUATE MODEL ------------------

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    """
    Trains a logistic regression model and evaluates its performance.
    Also plots and saves the ROC curve.

    Args:
        reg_rate (float): Regularization rate (inverse of C parameter in sklearn)
        X_train, X_test, y_train, y_test: Data splits
    """
    # Start a new MLflow run to track this experiment
    with mlflow.start_run():
        # Initialize Logistic Regression with inverse regularization strength
        model = LogisticRegression(C=1/reg_rate, solver="liblinear")
        
        # Fit the model to training data
        model.fit(X_train, y_train)

        # Predict class labels on test set
        y_pred = model.predict(X_test)
        
        # Evaluate accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # Get prediction probabilities for ROC curve
        y_scores = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_scores)
        print(f"ROC-AUC: {auc:.4f}")

        # Compute ROC curve values
        fpr, tpr, _ = roc_curve(y_test, y_scores)

        # Plot ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal baseline
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig("roc_curve.png")  # Save plot (automatically logged by MLflow)

# ------------------ PARSE COMMAND-LINE ARGUMENTS ------------------

def parse_args():
    """
    Parses CLI arguments for training the model.

    Returns:
        argparse.Namespace: Contains parsed values
    """
    parser = argparse.ArgumentParser()

    # Folder containing CSV files
    parser.add_argument('--training_data', type=str, required=True,
                        help='Path to folder containing training CSV files')

    # Regularization rate
    parser.add_argument('--reg_rate', type=float, default=0.1,
                        help='Regularization rate for logistic regression (default: 0.1)')

    return parser.parse_args()

# ------------------ SCRIPT ENTRY POINT ------------------

if __name__ == "__main__":
    args = parse_args()     # Parse CLI arguments
    main(args)              # Run training pipeline
    print("*" * 60)         # Print separator for log clarity
    print("\n\n")           # Add spacing between runs
    print("Training completed successfully!")  # Final message