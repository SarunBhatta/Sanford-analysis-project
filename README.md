# High Cost Claim Prediction Model

## Overview

This repository contains a Python script designed to predict "High Cost Claims" in a healthcare dataset using machine learning. The script preprocesses the data, engineers features, and trains a model using a combination of LinearSVC (Linear Support Vector Classification) and GridSearchCV for hyperparameter tuning. The model is evaluated using F1 score and balanced accuracy metrics.

The dataset used in this project is an Excel file (DSU-Dataset.xlsx), which contains patient information, claims data, and a target column (High Cost Claim) indicating whether a claim is high-cost or not.

## Key Features

1. Data Preprocessing:

Handles missing values using SimpleImputer.

Processes ICD-10 codes and modifiers using custom pipelines.

Encodes categorical features using OneHotEncoder and MultiLabelBinarizer.

1. Feature Engineering:

ICD-10 codes are preprocessed into lists of integers and binarized.

Modifiers are split into lists and binarized.

Categorical features (e.g., Gender, Marital Status, Ethnicity) are one-hot encoded.

1. Model Training:

Uses LinearSVC as the base model with class weighting to handle imbalanced data for its speed and competitive results.

Hyperparameters are tuned using GridSearchCV to optimize the F1 score.

The model is trained on a subset of the data, ensuring no data contamination by splitting patients into train and test sets.

1. Evaluation:

The model is evaluated on a held-out test set using F1 score and balanced accuracy.

1. Prediction:

Predicts "High Cost Claim" for unseen data and saves the results to a CSV file (sanford_results.csv).

## Requirements

To run this script, you need the following Python libraries:

scikit-learn

pandas

numpy

You can install the required libraries using:

```bash
pip install scikit-learn pandas numpy
```

## Usage

1. Prepare the Dataset:

Place the dataset (DSU-Dataset.xlsx) in the appropriate directory (e.g., ~/Downloads/).

2. Run the Script:

Execute the script using Jupyter.

3. Output:

The script will output the best hyperparameters, cross-validation score, and test set performance metrics (F1 score and balanced accuracy).

Predictions for the unseen data will be saved in sanford_results.csv.

## Customization

- Feature Engineering:

Modify the preprocess_icd10 and preprocess_modifiers functions to adjust the preprocessing logic.

Add or remove features in the ColumnTransformer as needed.

- Model Selection:

Replace LinearSVC with another classifier (e.g., RandomForestClassifier) by updating the pipeline.

- Hyperparameter Tuning:

Adjust the hyperparameter grid in GridSearchCV to explore different configurations.

## Performance Metrics

F1 Score: Measures the balance between precision and recall. It's slightly overfitting with:

- 98.10% training f1 score
- 75.89% validation f1 score
- 78.26% testing f1 score

Balanced Accuracy: Accounts for class imbalance by averaging the recall of each class.

## Notes

The script assumes that the High Cost Claim column is inflation-adjusted and does not use date features for prediction.

The dataset should be cleaned and formatted appropriately before running the script.
