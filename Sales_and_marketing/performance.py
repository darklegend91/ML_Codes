import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample data
data = {
    'Name': ['S1', 'S2', 'S3', 'S4', 'S5'],
    'Workshop 1': [10, 17, 25, 15, 15],
    'Workshop 2': [18, 23, 25, 19, 18],
    'Workshop 3': [13, 23, 22, 25, 23],
    'Workshop 4': [15, 20, 25, 23, 25]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate improvement for each topic
df['Improvement Topic 1'] = df['Workshop 2'] - df['Workshop 1']
df['Improvement Topic 2'] = df['Workshop 4'] - df['Workshop 3']

# Prepare data for clustering
X = df[['Improvement Topic 1', 'Improvement Topic 2']]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Save the DataFrame to an Excel file
df.to_excel('student_improvement_clusters.xlsx', index=False)

# Plot the clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'blue']
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Improvement Topic 1'], cluster_data['Improvement Topic 2'], 
                label=f'Cluster {cluster}', color=colors[cluster])

plt.title('Clustering of Student Improvements')
plt.xlabel('Improvement in Topic 1')
plt.ylabel('Improvement in Topic 2')
plt.legend()

# Save the plot as an image file
plt.savefig('student_improvement_clusters.png')

# Show the plot
plt.show()
