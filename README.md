# Cryptocurrency Clustering Analysis

This project aims to analyze and cluster cryptocurrencies using K-means clustering, employing data normalization and dimensionality reduction techniques such as Principal Component Analysis (PCA).
The Cryto_Clustering notebook file with interactive visuals can be easily viewed by clicking on this link: [[https://nbviewer.org/github.com/thaychansy/CryptoClustering/blob/main/Crypto_Clustering.ipynb](https://nbviewer.org/github/thaychansy/CryptoClustering/blob/main/Crypto_Clustering.ipynb)
](https://nbviewer.org/github/thaychansy/CryptoClustering/blob/main/Crypto_Clustering.ipynb)
## Table of Contents
- [Prepare the Data](#prepare-the-data)
- [Find the Best Value for k Using the Scaled DataFrame](#find-the-best-value-for-k-using-the-scaled-dataframe)
- [Cluster Cryptocurrencies with K-means Using the Scaled DataFrame](#cluster-cryptocurrencies-with-k-means-using-the-scaled-dataframe)
- [Optimize Clusters with Principal Component Analysis](#optimize-clusters-with-principal-component-analysis)
- [Find the Best Value for k Using the PCA DataFrame](#find-the-best-value-for-k-using-the-pca-dataframe)
- [Cluster Cryptocurrencies with K-means Using the PCA DataFrame](#cluster-cryptocurrencies-with-k-means-using-the-pca-dataframe)



## Prepare the Data
To normalize the data from the CSV file, we use the `StandardScaler()` module from `scikit-learn`. We create a DataFrame with the scaled data and set the "coin_id" from the original DataFrame as the index for the new DataFrame.

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
df_market_data = pd.read_csv(
    "Resources/crypto_market_data.csv",
    index_col="coin_id")

# Normalize data
price_change_scaled = StandardScaler().fit_transform(df_market_data[['price_change_percentage_24h', 'price_change_percentage_7d', 
                                                                    'price_change_percentage_14d', 'price_change_percentage_30d', 
                                                                    'price_change_percentage_60d', 'price_change_percentage_200d',
                                                                    'price_change_percentage_1y']])

# Create scaled DataFrame
price_change_scaled_df = pd.DataFrame(price_change_scaled, columns=['price_change_percentage_24h', 'price_change_percentage_7d', 
                                                                    'price_change_percentage_14d', 'price_change_percentage_30d', 
                                                                    'price_change_percentage_60d', 'price_change_percentage_200d',
                                                                    'price_change_percentage_1y'])
```

## Find the Best Value for k Using the Scaled DataFrame
Using the elbow method, we determine the best value for k through the following steps:

1. Create a list of k values from 1 to 11.
2. Compute the inertia for each value of k.
3. Plot the elbow curve.

```python
# Initialize lists
k_values = list(range(1, 12))
inertia = []

# Compute inertia for each k
for i in k:
    k_model = KMeans(n_clusters=i, random_state=0)
    k_model.fit(price_change_scaled_df)
    inertia.append(k_model.inertia_)

# Plot elbow curve
df_elbow.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)
```
![image](https://github.com/user-attachments/assets/4a70eaca-e259-4c5e-99ed-6d435815d1e6)


## Cluster Cryptocurrencies with K-means Using the Scaled DataFrame
Using the best k value, we cluster the cryptocurrencies as follows:

1. Initialize and fit the K-means model.
2. Predict clusters and add them to the DataFrame.
3. Create a scatter plot.

```python
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=3, random_state=1)

# Fit the K-Means model using the scaled data
model.fit(price_change_scaled_df)

# Predict the clusters to group the cryptocurrencies using the scaled data
k_3 = model.predict(price_change_scaled_df)

# Plot
price_change_scaled_predictions_df.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="crypto_segment",  # Color points by the cluster labels from K-Means
    hover_cols=["coinid"],  # Add the cryptocurrency name to hover information
    title="Crypto Segmentation based on K-Means Clustering (k=3)"
)
```
![image](https://github.com/user-attachments/assets/27f060ae-d946-4fe8-b003-918f66106933)

## Optimize Clusters with Principal Component Analysis
We perform PCA to reduce features to three principal components.

1. Fit PCA and retrieve explained variance.
2. Create a new DataFrame with PCA data.

```python
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)

# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
crypto_pca = pca.fit_transform(price_change_scaled)

# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_

# Create a new DataFrame with the PCA data.
crypto_pca_df = pd.DataFrame(
    crypto_pca,
    columns=["PCA1", "PCA2", "PCA3"]

# Set the coinid column as index
crypto_pca_df.set_index('coinid', inplace=True)
)
```
## Find the Best Value for k Using the PCA DataFrame
Use the elbow method on the PCA DataFrame as previously described to determine the best k value.

Best Value for k with PCA: (Include your findings here)Find the Best Value for k Using the PCA DataFrame
Use the elbow method on the PCA DataFrame as previously described to determine the best k value.

```python
# Create a list with the number of k-values from 1 to 11
k = list(range(1, 12))

# Create a list with the number of k-values from 1 to 11
k = list(range(1, 12))

# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
for i in k:
    k_model = KMeans(n_clusters=i, random_state=0)
    k_model.fit(crypto_pca_df)
    inertia.append(k_model.inertia_)
```

```python
# Plot
df_elbow_pca.hvplot.line(
    x="k",
    y="inertia",
    title="Elbow Curve PCA",
    xticks=k
)
```
![image](https://github.com/user-attachments/assets/a3487987-351a-43bb-a2e8-65378f2bf3eb)

## Cluster Cryptocurrencies with K-means Using the PCA DataFrame
Repeat the clustering process using the PCA DataFrame.

Initialize K-means with the best k value from PCA.
1. Fit and predict clusters.
2. Create a scatter plot.

```python

# Initialize K-means with best k from PCA
model = KMeans(n_clusters=4, random_state=0)
model.fit(crypto_pca_df)

# Predict the clusters to group the cryptocurrencies using the PCA data
k_4 = model.predict(crypto_pca_df)

# Create a copy of the DataFrame with the PCA data
crypto_pca_predictions_df = crypto_pca_df.copy()

# Add a new column to the DataFrame with the predicted clusters
crypto_pca_predictions_df["cryto_segments"] = k_4

# Plot
crypto_pca_predictions_df.hvplot.scatter(
    x="PCA1",
    y="PCA2",
    by="cryto_segments",
    hover_cols=["coinid"],  # Add the cryptocurrency name to hover information
    title="PCA Crypto Segmentation based on K-Means Clustering (k=4)"
)
```
![image](https://github.com/user-attachments/assets/e3678883-2d75-4e5b-afea-b597f62d6c1d)


## Conclusion
This analysis provides insights into the clustering of cryptocurrencies using both the scaled data and the PCA-transformed data. The results can help in identifying trends and making informed investment decisions.

![image](https://github.com/user-attachments/assets/41e65067-d7df-4e5c-8309-5d6e486e791d)

![image](https://github.com/user-attachments/assets/edb8c288-068f-4a50-9e06-5ab1f1ea44bf)

## Contact

Thay Chansy - [@thaychansy](https://twitter.com/thaychansy) - or thay.chansy@gmail.com


Please visit my Portfolio Page: thaychansy.github.io (https://thaychansy.github.io/)
