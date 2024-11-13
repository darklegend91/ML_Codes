import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns

# Load the data from the Excel file
file_path = 'data.xlsx'

# Load activity data
activity_data = pd.read_excel(file_path, sheet_name='Activity_data')
attendance_data = pd.read_excel(file_path, sheet_name='attendance')
marks_data = pd.read_excel(file_path, sheet_name='Introduction to sales and marketing post test')

# Merge attendance and marks data on student ID or name (assume 'Name' column exists)
merged_data = pd.merge(attendance_data, marks_data, on='Name')
merged_data = pd.merge(merged_data, activity_data, how='left', on='Name')

# Preprocess the data for modeling
# Encode categorical columns if necessary (e.g., gender, activity category)
merged_data['activity_category'] = merged_data['activity_category'].astype('category').cat.codes

# Features: attendance and initial marks
X = merged_data[['attendance', 'marks']]
y = merged_data['marks']  # Replace with actual column if different

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on all activities in the activity data
activity_data['Predicted Marks'] = model.predict(activity_data[['attendance', 'marks']])

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Plot student performance over activities
plt.figure(figsize=(10, 6))
sns.lineplot(data=merged_data, x='date', y='marks', hue='Name')
plt.title('Student Performance Over Activities')
plt.xlabel('Date')
plt.ylabel('Marks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot performance by activity category
plt.figure(figsize=(10, 6))
sns.boxplot(data=merged_data, x='activity_category', y='marks')
plt.title('Performance of Students in Each Category')
plt.xlabel('Activity Category')
plt.ylabel('Marks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot performance comparison of both genders
plt.figure(figsize=(10, 6))
sns.boxplot(data=merged_data, x='gender', y='marks')
plt.title('Performance Comparison by Gender')
plt.xlabel('Gender')
plt.ylabel('Marks')
plt.tight_layout()
plt.show()

# Save the predictions and data to a new Excel file
output_file = 'student_performance_analysis.xlsx'
activity_data.to_excel(output_file, index=False)
print(f'Results saved to {output_file}')
