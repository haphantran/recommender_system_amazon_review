# Item-Based Collaborative Filtering (ItemKNN) - Training and Evaluation Script
# Import necessary libraries
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os

# Load dataset from Hugging Face
print("Loading dataset for Item-Based Collaborative Filtering from Hugging Face...")
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", split="full", trust_remote_code=True)

# Convert to a pandas DataFrame and reduce the dataset size for memory efficiency
print("Converting dataset to pandas DataFrame and reducing size...")
data = dataset.to_pandas()
reduced_data = data.sample(frac=0.1, random_state=42)  # Reduce to 10% of the data for memory efficiency

# Load reduced dataset into Surprise format
print("Loading reduced dataset into Surprise format...")
reader = Reader(rating_scale=(reduced_data['rating'].min(), reduced_data['rating'].max()))
dataset = Dataset.load_from_df(reduced_data[['user_id', 'asin', 'rating']], reader)

# Train-test split
print("Splitting data for Item-Based Collaborative Filtering model...")
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
print("Data split completed.")

# Item-Based Collaborative Filtering with 'cosine' similarity
print("Training Item-Based Collaborative Filtering model...")
sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based similarity
}

item_knn = KNNBasic(sim_options=sim_options)
item_knn.fit(trainset)
predictions_item_knn = item_knn.test(testset)
print("Item-Based Collaborative Filtering model trained.")

# Evaluate the model
print("Evaluating Item-Based Collaborative Filtering model...")
rmse_item_knn = accuracy.rmse(predictions_item_knn)
mae_item_knn = accuracy.mae(predictions_item_knn)

# Convert predictions to binary for classification metrics
y_true = [int(pred.r_ui >= 3) for pred in predictions_item_knn]  # Assume rating >= 3 is positive
y_pred = [int(pred.est >= 3) for pred in predictions_item_knn]

precision_item_knn = precision_score(y_true, y_pred, zero_division=1)
recall_item_knn = recall_score(y_true, y_pred, zero_division=1)
f1_item_knn = f1_score(y_true, y_pred, zero_division=1)
auc_item_knn = roc_auc_score(y_true, y_pred)

# Save evaluation results to a log file in a consistent format
results = {
    'Model': 'ItemKNN',
    'RMSE': rmse_item_knn,
    'MAE': mae_item_knn,
    'Precision': precision_item_knn,
    'Recall': recall_item_knn,
    'F1': f1_item_knn,
    'AUC': auc_item_knn
}

results_folder = "./results/"
os.makedirs(results_folder, exist_ok=True)
with open(os.path.join(results_folder, 'item_knn_results.txt'), 'w') as f:
    f.write(str(results))

print("Item-Based Collaborative Filtering evaluation completed and results saved.")
