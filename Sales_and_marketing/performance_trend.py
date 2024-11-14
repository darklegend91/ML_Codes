import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data from the Excel file
file_path = 'student_activity_predictions.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Pivot the data to have students as rows, activities as columns, and predicted marks as values
pivot_data = data.pivot_table(index='Name', columns='Activity', values='Predicted Marks')

# Fill any missing values (if any) with the mean value of each column
pivot_data = pivot_data.fillna(pivot_data.mean())

# Standardize the data (important for clustering)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_data)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # You can change n_clusters based on your requirement
pivot_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Plot the trends for each student
plt.figure(figsize=(12, 8))

for student in pivot_data.index:
    plt.plot(pivot_data.columns, pivot_data.loc[student], label=student)

plt.xlabel('Activity')
plt.ylabel('Predicted Marks')
plt.title('Performance Trend of Each Student Across Activities')
plt.xticks(rotation=90)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# Plot the clusters (optional: you can plot clusters in a 2D space if you reduce dimensions)
plt.figure(figsize=(10, 6))
plt.scatter(pivot_data.index, [0]*len(pivot_data), c=pivot_data['Cluster'], cmap='viridis')
plt.xlabel('Student')
plt.ylabel('Cluster')
plt.title('Student Clusters Based on Performance Trends')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Save the cluster data to a new Excel file
pivot_data.to_excel('student_activity_clusters.xlsx')
