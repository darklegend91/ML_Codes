import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Excel file
file_path = 'data.xlsx'  # Replace with the correct file path
attendance_data = pd.read_excel(file_path, sheet_name='attendance')

# Convert attendance data to numerical format (Present = 1, Absent = 0)
attendance_data.replace({'Present': 1, 'ABSENT': 0, 'Absent': 0}, inplace=True)

# Extract student names and genders for identification
student_info = attendance_data[['Name', 'Gender']]

# Calculate attendance statistics
attendance_counts = attendance_data.iloc[:, 3:].sum(axis=1)
total_days = attendance_data.iloc[:, 3:].shape[1]

# Categorize students based on attendance
attendance_data['Attendance Rate'] = attendance_counts / total_days

def categorize_student(rate):
    if rate > 0.9:
        return 'Consistently Present'
    elif rate < 0.5:
        return 'Consistently Absent'
    else:
        return 'Fluctuating'

attendance_data['Category'] = attendance_data['Attendance Rate'].apply(categorize_student)

# Merge student info with the analysis
final_data = pd.concat([student_info, attendance_data[['Attendance Rate', 'Category']]], axis=1)

# Plotting student performance over attendance
plt.figure(figsize=(12, 6))
sns.barplot(x='Name', y='Attendance Rate', data=final_data, hue='Category')
plt.title('Student Attendance Analysis')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the overall distribution of categories
plt.figure(figsize=(8, 6))
sns.countplot(x='Category', data=final_data)
plt.title('Overall Attendance Category Distribution')
plt.tight_layout()
plt.show()

# Save the results to an Excel file
output_file = 'attendance_analysis.xlsx'
final_data.to_excel(output_file, index=False)
print(f'Results saved to {output_file}')
