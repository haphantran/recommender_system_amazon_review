# Setup and Data Loading
# First, let's start by setting up the environment and loading the dataset. This involves downloading the dataset, preprocessing it, and creating an environment for modeling.

# Install Required Libraries
# For this assignment, you will need several libraries:
# - `pandas`, `numpy` for data handling
# - `scikit-learn` for preprocessing and evaluation
# - `scikit-surprise` for baseline models
# - `recbole` for running advanced recommendation models
# - `datasets` from Hugging Face for loading the dataset
# - `tensorflow` for deep learning models

# Run the following command in your terminal to install the libraries
# !pip install pandas numpy scikit-learn scikit-surprise recbole datasets tensorflow

# Data Loading and Preprocessing
# Let's load the data using the Hugging Face `datasets` library and handle basic preprocessing.

import pandas as pd
import numpy as np
from datasets import load_dataset
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder

print("Loading dataset...")
# Load dataset
# Replace 'raw_review_All_Beauty' with the desired dataset category
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", split="full", trust_remote_code=True)

# Convert to a pandas DataFrame
data = dataset.to_pandas()
print("Dataset loaded successfully.")

# Basic Preprocessing
print("Starting data preprocessing...")
# Convert nested columns (if any) to a hashable format, such as strings
data = data.applymap(lambda x: str(x) if isinstance(x, (list, dict, np.ndarray)) else x)

# Printing basic statistics
print("Basic statistics of the dataset:")
print(data.describe())
print(data.info())

# Splitting data into train-test sets
from sklearn.model_selection import train_test_split

print("Splitting data into train and test sets...")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print("Data split completed.")

# Implement Baseline Models
# Now, let’s start with some baseline recommendation models using the `Surprise` library from `scikit-surprise`.

# Install and Setup the Surprise Library
# `scikit-surprise` is great for collaborative filtering methods.

from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

print("Loading dataset into Surprise format...")
# Load dataset into Surprise format
reader = Reader(rating_scale=(data['rating'].min(), data['rating'].max()))
dataset = Dataset.load_from_df(data[['user_id', 'asin', 'rating']], reader)
print("Dataset loaded into Surprise format.")

# Train-test split
print("Splitting data for Surprise model...")
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
print("Data split for Surprise model completed.")

# Item-Based Collaborative Filtering (ItemKNN)
print("Training Item-Based Collaborative Filtering model...")
# Item-Based Collaborative Filtering
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
print(f"ItemKNN - RMSE: {rmse_item_knn}, MAE: {mae_item_knn}")

# User-Based Collaborative Filtering (UserKNN) with reduced data and different similarity metric
print("Training User-Based Collaborative Filtering model with reduced data and different similarity metric...")
# Reduce dataset size for UserKNN to avoid memory issues
reduced_data = data.sample(frac=0.1, random_state=42)

# Load reduced dataset into Surprise format
reader_reduced = Reader(rating_scale=(reduced_data['rating'].min(), reduced_data['rating'].max()))
dataset_reduced = Dataset.load_from_df(reduced_data[['user_id', 'asin', 'rating']], reader_reduced)

# Train-test split for reduced data
trainset_reduced, testset_reduced = train_test_split(dataset_reduced, test_size=0.2, random_state=42)

# User-Based Collaborative Filtering with 'pearson_baseline' similarity
sim_options = {
    'name': 'pearson_baseline',
    'user_based': True  # User-based similarity
}

user_knn = KNNBasic(sim_options=sim_options)
user_knn.fit(trainset_reduced)
predictions_user_knn = user_knn.test(testset_reduced)
print("User-Based Collaborative Filtering model trained.")

# Evaluate the model
print("Evaluating User-Based Collaborative Filtering model...")
rmse_user_knn = accuracy.rmse(predictions_user_knn)
mae_user_knn = accuracy.mae(predictions_user_knn)
print(f"UserKNN - RMSE: {rmse_user_knn}, MAE: {mae_user_knn}")

# Matrix Factorization (SVD)
print("Training Matrix Factorization (SVD) model...")
# Matrix Factorization using SVD
svd = SVD()
svd.fit(trainset)
predictions_svd = svd.test(testset)
print("Matrix Factorization (SVD) model trained.")

# Evaluate the model
print("Evaluating Matrix Factorization (SVD) model...")
rmse_svd = accuracy.rmse(predictions_svd)
mae_svd = accuracy.mae(predictions_svd)
print(f"SVD - RMSE: {rmse_svd}, MAE: {mae_svd}")

# Advanced Model - Using RecBole
# For an advanced model, you can use the `RecBole` library for more sophisticated recommendations.

from recbole.quick_start import run_recbole

# Define the model configuration
config_dict = {
    'dataset': 'YourDataset',
    'model': 'NFM',  # Neural Factorization Machine as an example
    'config_files': ['recbole_config_file.yaml']  # Path to your RecBole config file
}

print("Running advanced model using RecBole...")
# Run the model
run_recbole(config_dict=config_dict)
print("RecBole model run completed.")

# Deep Learning Model with TensorFlow
# Adding a simple deep learning model using TensorFlow
print("Training Deep Learning model using TensorFlow...")

# Encoding user_id and item_id
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

train_data['user_id'] = user_encoder.fit_transform(train_data['user_id'])
train_data['asin'] = item_encoder.fit_transform(train_data['asin'])
test_data['user_id'] = user_encoder.transform(test_data['user_id'])
test_data['asin'] = item_encoder.transform(test_data['asin'])

# Defining model parameters
n_users = len(user_encoder.classes_)
n_items = len(item_encoder.classes_)
embedding_size = 50

# Building the model
model = Sequential()
model.add(Embedding(input_dim=n_users, output_dim=embedding_size, input_length=1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Preparing training data
x_train = [train_data['user_id'], train_data['asin']]
y_train = train_data['rating'].astype(np.float32)

x_test = [test_data['user_id'], test_data['asin']]
y_test = test_data['rating'].astype(np.float32)

# Training the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluating the model
print("Evaluating Deep Learning model using TensorFlow...")
loss, mae = model.evaluate(x_test, y_test)
print(f"Deep Learning Model - Loss: {loss}, MAE: {mae}")

# Evaluation Metrics
# After running all the models, we need to evaluate them using RMSE, MAE, Precision, and AUC.

# Displaying Results in a Table
# Below is an example of how to store results for comparison.

print("Compiling results...")
results = {
    'Model': ['ItemKNN', 'UserKNN', 'SVD', 'RecBole-NFM', 'Deep Learning (TensorFlow)'],
    'RMSE': [rmse_item_knn, rmse_user_knn, rmse_svd, 'N/A', 'N/A'],
    'MAE': [mae_item_knn, mae_user_knn, mae_svd, 'N/A', mae],
    'Precision@10': ['N/A', 'N/A', 'N/A', 'Generated in RecBole log', 'N/A'],
    'AUC': ['N/A', 'N/A', 'N/A', 'Generated in RecBole log', 'N/A']
}

results_df = pd.DataFrame(results)
print("Results compiled successfully.")
print(results_df)

# Next Steps
# 1. **Run the Code**: Set up your environment and run the provided code snippets step-by-step.
# 2. **Choose an Advanced Model**: You can replace NFM in RecBole with other models like NeuMF, GCN, etc.
# 3. **Write the Report**: Include all results, descriptions, and evaluations as mentioned in your assignment.
