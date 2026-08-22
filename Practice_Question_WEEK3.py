#-----------------------Practice Question WEEK3-----------------------#


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv("Iris.csv")
X = df.iloc[:, 1:5]

# Q1. Apply K-Means and print cluster centers

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
print("Cluster Centers:")
print(kmeans.cluster_centers_)


# Q2. Visualize clusters


plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris K-Means Clusters")
plt.show()


# Q3. Apply PCA to reduce dataset to 2D


X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("PCA Data:")
print(X_pca)


# Q4. Find explained variance ratio


print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)