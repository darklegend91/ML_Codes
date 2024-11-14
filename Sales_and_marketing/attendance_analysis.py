import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

# Load the data from the Excel file
file_path = 'Sales_and_marketing/data.xlsx'  # Replace with the correct file path
attendance_data = pd.read_excel(file_path, sheet_name='attendance')

# Convert attendance data to numerical format (Present = 1, Absent = 0)
attendance_data.replace({'P': 1, 'A': 0}, inplace=True)

# Extract student names, genders, and attendance columns
student_info = attendance_data[['Name', 'Gender']]
attendance_columns = attendance_data.columns[3:]
attendance_only = attendance_data[attendance_columns]

# Step 1: Clustering students based on their attendance
kmeans = KMeans(n_clusters=3, random_state=42)
attendance_data['Cluster'] = kmeans.fit_predict(attendance_only)

# Merge clustering results with student information
final_data = pd.concat([student_info, attendance_data[['Cluster']]], axis=1)

# Save clustering results to Excel
output_file = 'attendance_clustering_results.xlsx'
final_data.to_excel(output_file, index=False)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=attendance_data, x=attendance_only.mean(axis=1), y=attendance_only.std(axis=1), hue='Cluster', palette='viridis')
plt.title('Clustering of Students Based on Attendance')
plt.xlabel('Average Attendance')
plt.ylabel('Attendance Variation')
plt.tight_layout()
plt.savefig('attendance_clustering.png')
plt.show()

# Step 2: Plot attendance of each student over activities
plt.figure(figsize=(14, 8))
for index, row in attendance_data.iterrows():
    plt.plot(attendance_columns, row[attendance_columns], marker='o', label=row['Name'])

plt.title('Attendance Over Activities for Each Student')
plt.xlabel('Activity/Date')
plt.ylabel('Attendance (1 = Present, 0 = Absent)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig('student_attendance_over_activities.png')
plt.show()

# Step 3: Plot gender-wise attendance trend
gender_trends = attendance_data.groupby('Gender')[attendance_columns].mean().transpose()

plt.figure(figsize=(12, 6))
sns.lineplot(data=gender_trends)
plt.title('Gender-wise Attendance Trend Comparison')
plt.xlabel('Activity/Date')
plt.ylabel('Average Attendance (Proportion)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('gender_attendance_trend.png')
plt.show()

print(f'Clustering and trend plots saved as images and data saved to {output_file}')
