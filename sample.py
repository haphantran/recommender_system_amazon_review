

# Basic Preprocessing
# Dropping duplicates
data = data.drop_duplicates()

# Handling missing values (depends on the dataset, here dropping rows with NaN)
data = data.dropna()

# Printing basic statistics
print(data.describe())
print(data.info())

# Splitting data into train-test sets
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load dataset into Surprise format
reader = Reader(rating_scale=(data['rating'].min(), data['rating'].max()))
dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)


# Item-Based Collaborative Filtering
sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based similarity
}

item_knn = KNNBasic(sim_options=sim_options)
item_knn.fit(trainset)
predictions_item_knn = item_knn.test(testset)

# Evaluate the model
rmse_item_knn = accuracy.rmse(predictions_item_knn)
mae_item_knn = accuracy.mae(predictions_item_knn)


# User-Based Collaborative Filtering
sim_options = {
    'name': 'cosine',
    'user_based': True  # User-based similarity
}

user_knn = KNNBasic(sim_options=sim_options)
user_knn.fit(trainset)
predictions_user_knn = user_knn.test(testset)

# Evaluate the model
rmse_user_knn = accuracy.rmse(predictions_user_knn)
mae_user_knn = accuracy.mae(predictions_user_knn)


# Matrix Factorization using SVD
svd = SVD()
svd.fit(trainset)
predictions_svd = svd.test(testset)

# Evaluate the model
rmse_svd = accuracy.rmse(predictions_svd)
mae_svd = accuracy.mae(predictions_svd)


from recbole.quick_start import run_recbole

# Define the model configuration
config_dict = {
    'dataset': 'YourDataset',
    'model': 'NFM',  # Neural Factorization Machine as an example
    'config_files': ['recbole_config_file.yaml']  # Path to your RecBole config file
}

# Run the model
run_recbole(config_dict=config_dict)


results = {
    'Model': ['ItemKNN', 'UserKNN', 'SVD', 'RecBole-NFM'],
    'RMSE': [rmse_item_knn, rmse_user_knn, rmse_svd, 'N/A'],
    'MAE': [mae_item_knn, mae_user_knn, mae_svd, 'N/A'],
    'Precision@10': ['N/A', 'N/A', 'N/A', 'Generated in RecBole log'],
    'AUC': ['N/A', 'N/A', 'N/A', 'Generated in RecBole log']
}

results_df = pd.DataFrame(results)
print(results_df)
