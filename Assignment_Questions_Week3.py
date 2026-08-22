#----------------------------------Assignment Questions Week 3---------------------------#

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv("Iris.csv")
X = df.iloc[:, 1:5]

# Q1. Perform K-Means on Iris dataset and visualize clusters

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
print("Cluster Centers:")
print(kmeans.cluster_centers_)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering")
plt.show()


# Q2. Apply PCA to reduce dataset dimensions


X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("PCA Result:")
print(X_pca)
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)