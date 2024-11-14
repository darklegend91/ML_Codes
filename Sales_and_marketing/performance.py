import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the attendance and marks data
attendance_file_path = 'Sales_and_marketing/data.xlsx'  # Replace with your file path
attendance_data = pd.read_excel(attendance_file_path, sheet_name='attendance')  # Adjust the sheet name as necessary

marks_file_path = 'Sales_and_marketing/data.xlsx'  # Replace with your file path
marks_data = pd.read_excel(marks_file_path, sheet_name='mark')  # Adjust the sheet name as necessary

# Preprocess the data: Convert attendance to numerical format (e.g., P = 1, A = 0)
attendance_data.replace({'P': 1, 'A': 0}, inplace=True)

# Ensure matching of student names and link data
merged_data = pd.merge(attendance_data, marks_data[['Name', 'marks']], on='Name', how='inner')

# Identify columns for attendance data (excluding metadata)
attendance_columns = attendance_data.columns[3:]  # Assuming first 3 columns are metadata

# Prepare features (X) and target (y)
X = merged_data[attendance_columns]
y = merged_data['marks']

# Convert all column names in X to strings
X.columns = X.columns.astype(str)

# Handle missing values in the data by imputing with the median value
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)  # Impute missing values in X

# Train a regression model
model = LinearRegression()
model.fit(X_imputed, y)

# Predict marks for all activities
predicted_marks = model.predict(X_imputed)

# List of activities and dates
activities = [
    ("Introduction MS Word and Sales Marketing", "17-05-2024"),
    ("Typing Activity and Insert Header Footer in MS Word", "12-07-2024"),
    ("Creating Header and Formulas in MS Excel", "24-07-2024"),
    ("Fundamental Features and Functionalities of Microsoft Excel", "26-07-2024"),
    ("Creating a Pie and Bar Chart in Excel Activity Sheet", "02-08-2024"),
    ("Activity Sheet (About Summer Olympics PPT) MS PowerPoint", "21-08-2024"),
    ("MS Office (Word, Excel, and PowerPoint) Revision", "23-08-2024"),
    ("Revision of Sales Process", "06-09-2024"),
    ("Revision Test Exam", "11-09-2024"),
    ("Revision of All Aspects and Steps of the Sales Process", "25-09-2024"),
    ("Explained the 4 Ps of the Marketing Concept", "27-09-2024"),
    ("Revision of Sales Marketing Process", "04-10-2024"),
    ("Sales and Marketing Test", "09-10-2024")
]

# Prepare data for each student on each date
predictions_data = []

for idx, student in merged_data.iterrows():
    for activity, date in activities:
        student_prediction = {
            'Name': student['Name'],
            'Activity': activity,
            'Date': date,
            'Predicted Marks': predicted_marks[idx]
        }
        predictions_data.append(student_prediction)

# Convert the predictions data to a DataFrame
predictions_df = pd.DataFrame(predictions_data)

# Save the predictions for each student and activity to Excel
output_file = 'student_activity_predictions.xlsx'
predictions_df.to_excel(output_file, index=False)

# Plot predicted marks for all activities
plt.figure(figsize=(10, 5))
plt.bar(predictions_df['Activity'], predictions_df['Predicted Marks'], color='skyblue')
plt.xlabel('Activity')
plt.ylabel('Predicted Marks')
plt.title('Predicted Marks for Activities')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('activity_predictions_plot.png')
plt.show()

print(f'Predicted marks for each student and activity have been saved to {output_file} and the plot as an image.')
