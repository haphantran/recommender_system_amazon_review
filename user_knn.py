# User-Based Collaborative Filtering (UserKNN) - Training and Evaluation Script
# Import necessary libraries
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os

# Load the reduced dataset from Hugging Face
print("Loading reduced dataset for User-Based Collaborative Filtering from Hugging Face...")
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", split="full", trust_remote_code=True)

# Convert to a pandas DataFrame and reduce the dataset size
print("Converting dataset to pandas DataFrame and reducing size...")
data = dataset.to_pandas()
reduced_data = data.sample(frac=0.1, random_state=42)  # Reduce to 10% of the data for memory efficiency

# Load reduced dataset into Surprise format
reader = Reader(rating_scale=(reduced_data['rating'].min(), reduced_data['rating'].max()))
dataset = Dataset.load_from_df(reduced_data[['user_id', 'asin', 'rating']], reader)

# Train-test split
print("Splitting data for User-Based Collaborative Filtering model...")
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
print("Data split completed.")

# User-Based Collaborative Filtering with 'pearson_baseline' similarity
print("Training User-Based Collaborative Filtering model...")
sim_options = {
    'name': 'pearson_baseline',
    'user_based': True  # User-based similarity
}

user_knn = KNNBasic(sim_options=sim_options)
user_knn.fit(trainset)
predictions_user_knn = user_knn.test(testset)
print("User-Based Collaborative Filtering model trained.")

# Evaluate the model
print("Evaluating User-Based Collaborative Filtering model...")
rmse_user_knn = accuracy.rmse(predictions_user_knn)
mae_user_knn = accuracy.mae(predictions_user_knn)

# Convert predictions to binary for classification metrics
y_true = [int(pred.r_ui >= 3) for pred in predictions_user_knn]  # Assume rating >= 3 is positive
y_pred = [int(pred.est >= 3) for pred in predictions_user_knn]

precision_user_knn = precision_score(y_true, y_pred, zero_division=1)
recall_user_knn = recall_score(y_true, y_pred, zero_division=1)
f1_user_knn = f1_score(y_true, y_pred, zero_division=1)
auc_user_knn = roc_auc_score(y_true, y_pred)

# Save evaluation results to a log file in a consistent format
results = {
    'Model': 'UserKNN',
    'RMSE': rmse_user_knn,
    'MAE': mae_user_knn,
    'Precision': precision_user_knn,
    'Recall': recall_user_knn,
    'F1': f1_user_knn,
    'AUC': auc_user_knn
}

results_folder = "./results/"
os.makedirs(results_folder, exist_ok=True)
with open(os.path.join(results_folder, 'user_knn_results.txt'), 'w') as f:
    f.write(str(results))

print("User-Based Collaborative Filtering evaluation completed and results saved.")
