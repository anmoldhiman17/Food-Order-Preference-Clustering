# ------------------------------
# 📌 Step 1: Import Libraries
# ------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------
# 📌 Step 2: Load Dataset
# ------------------------------
df = pd.read_csv("archive (9).zip")

# ------------------------------
# 📌 Step 3: Feature Engineering
# ------------------------------
# Calculate per customer metrics:
# - Order_Frequency = total orders
# - Spend_Amount = total spend
# - Cuisine_Type = most frequent restaurant

user_orders = df.groupby('Customer ID').agg({
    'Order ID': 'count',
    'Total': 'sum',
    'Restaurant name': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
}).reset_index()

user_orders.rename(columns={
    'Order ID': 'Order_Frequency',
    'Total': 'Spend_Amount',
    'Restaurant name': 'Cuisine_Type'
}, inplace=True)

print("\n✅ Processed Data Sample:\n", user_orders.head())

# ------------------------------
# 📌 Step 4: One-Hot Encode Cuisine_Type
# ------------------------------
encoder = OneHotEncoder(handle_unknown='ignore')
cuisine_encoded = encoder.fit_transform(user_orders[['Cuisine_Type']]).toarray()

# ✅ Fixed: No manual column name, auto-generate names
cuisine_df = pd.DataFrame(cuisine_encoded, columns=encoder.get_feature_names_out())

df_encoded = pd.concat([user_orders[['Order_Frequency','Spend_Amount']], cuisine_df], axis=1)

# ------------------------------
# 📌 Step 5: Scale Data
# ------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

# ------------------------------
# 📌 Step 6: PCA for Dim Reduction
# ------------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# ------------------------------
# 📌 Step 7: Elbow Method
# ------------------------------
wcss = []
for i in range(2,10):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(2,10), wcss, marker='o')
plt.title("Elbow Method - Optimal Clusters")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.show()

# ------------------------------
# 📌 Step 8: Apply KMeans
# ------------------------------
optimal_k = 4   # 👈 Elbow plot dekh ke change karna
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pca_data)

user_orders['Cluster'] = clusters

# ------------------------------
# 📌 Step 9: Evaluate
# ------------------------------
score = silhouette_score(pca_data, clusters)
print("\n✅ Silhouette Score:", score)

# ------------------------------
# 📌 Step 10: Visualize Clusters
# ------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=clusters, palette="Set2", s=80)
plt.title("Food Order Preference Clusters")
plt.show()

# ------------------------------
# 📌 Step 11: Cluster Summary
# ------------------------------
cluster_summary = user_orders.groupby('Cluster')[['Order_Frequency','Spend_Amount']].mean()
print("\n✅ Cluster Summary:\n", cluster_summary)
