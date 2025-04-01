import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os

# Load data from provided file path
file_path = r"investment_allocation.csv"
data = pd.read_csv(file_path)

# Extract sector columns and prepare training data
sectors = [
    'Artificial Intelligence',
    'Renewable Energy',
    'Technology',
    'Pharmaceuticals',
    'Healthcare',
    'E-commerce & Digital Economy',
    'Automotive (Including EVs)'
]

X = data[sectors].values
scaler = StandardScaler().fit(X)

# Create and train K-means model
kmeans = KMeans(n_clusters=5, random_state=42).fit(scaler.transform(X))

# Create age group to cluster mapping
age_groups = ['20-29', '30-39', '40-49', '50-59', '60+']
data['Cluster'] = kmeans.labels_
age_cluster_map = dict(zip(age_groups, kmeans.labels_))

# Save model in the "Project expo" folder
output_path = os.path.join(r"C:\Users\adity\PycharmProjects\IETproj", "investment_clustering_model.pkl")
with open(output_path, 'wb') as f:
    pickle.dump({
        'model': kmeans,
        'scaler': scaler,
        'sectors': sectors,
        'age_map': age_cluster_map
    }, f)

print(f"Model saved successfully at {output_path}")

def allocate_investment(age, amount):
    """Allocate investments using trained model with proper sector names."""
    # Determine age group based on input age
    if 20 <= age < 30: group = '20-29'
    elif 30 <= age < 40: group = '30-39'
    elif 40 <= age < 50: group = '40-49'
    elif 50 <= age < 60: group = '50-59'
    else: group = '60+'

    # Load model artifacts from saved file
    with open(output_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    # Get allocation pattern for the given age group
    cluster_idx = artifacts['age_map'][group]
    allocation = artifacts['scaler'].inverse_transform(
        [artifacts['model'].cluster_centers_[cluster_idx]]
    )[0]
    
    # Calculate percentages and allocate funds proportionally
    total = allocation.sum()
    return {
        sector: round((value / total) * amount, 2)
        for sector, value in zip(artifacts['sectors'], allocation)
    }

# Example usage
print(allocate_investment(35, 50000))  # For a 35-year-old investing ₹50,000
print(allocate_investment(62, 100000)) # For a 62-year-old investing ₹100,000
print(allocate_investment(21, 100000))
print(allocate_investment(25, 100000))