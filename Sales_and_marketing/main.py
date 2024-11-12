#Here we have used K _means clustering to group the students into different categories based on their performance. The clusters are based on parameters like pre-test score, post-test score, attendance, and improvement rate, categorizing the students into five groups for targeted attention.

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Generate random data for students
np.random.seed(42)

# Number of students
n_students = 32

# Random data for Pre-test, Post-test, Attendance, and Improvement Rate
attendance = np.random.randint(70, 100, size=n_students)  # Attendance percentage
pre_test_score = np.random.randint(5, 15, size=n_students)  # Pre-test score (out of 20)
post_test_score = pre_test_score + np.random.randint(3, 10, size=n_students)  # Post-test score
improvement_rate = post_test_score - pre_test_score  # Improvement rate (change in score)

# Create DataFrame
data = pd.DataFrame({
    'Attendance': attendance,
    'Pre-Test Score': pre_test_score,
    'Post-Test Score': post_test_score,
    'Improvement Rate': improvement_rate
})

# Step 2: Perform K-Means Clustering
# Features: Attendance, Pre-Test Score, Post-Test Score, Improvement Rate
features = data[['Attendance', 'Pre-Test Score', 'Post-Test Score', 'Improvement Rate']]

# K-Means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
data['Category'] = kmeans.fit_predict(features)

# Step 3: Save the results in an Excel sheet
output_file = "student_performance_clusters.xlsx"
data.to_excel(output_file, index=False)

# Step 4: Visualizing the clusters
plt.figure(figsize=(8,6))
plt.scatter(data['Pre-Test Score'], data['Improvement Rate'], c=data['Category'], cmap='viridis', s=100)
plt.title('K-Means Clustering of Students')
plt.xlabel('Pre-Test Score')
plt.ylabel('Improvement Rate')
plt.colorbar(label='Cluster')
plt.show()

output_file
