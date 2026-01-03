import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# STEP 1: Load Dataset
data = pd.read_csv("students.csv")
print("Dataset Loaded:\n", data)

# STEP 2: Separate Inputs & Output
X = data[['StudyHours', 'Attendance']]
y = data['Marks']

# STEP 3: Split Data (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 5: Predict Test Set
predictions = model.predict(X_test)
print("\nPredicted Marks:", predictions)
print("Actual Marks:", y_test.values)

# STEP 6: Evaluate Model
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# STEP 7: Predict for New Student (User Input)
study_hours = float(input("\nEnter study hours: "))
attendance = float(input("Enter attendance percentage: "))

new_student = [[study_hours, attendance]]
predicted_marks = model.predict(new_student)
print("Predicted Student Marks:", round(predicted_marks[0], 2))

# STEP 8: Visualization

# 2D Scatter: Study Hours vs Marks
plt.scatter(data['StudyHours'], data['Marks'], color='blue')
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# 2D Scatter: Attendance vs Marks
plt.scatter(data['Attendance'], data['Marks'], color='green')
plt.title("Attendance vs Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Marks")
plt.show()

# Predicted vs Actual
plt.scatter(y_test, predictions, color='red')
plt.title("Actual vs Predicted Marks")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')  # diagonal
plt.show()

# 3D Scatter: Study Hours + Attendance vs Marks
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['StudyHours'], data['Attendance'], data['Marks'], color='purple', s=50)
ax.set_xlabel('Study Hours')
ax.set_ylabel('Attendance (%)')
ax.set_zlabel('Marks')
ax.set_title('3D Scatter: StudyHours + Attendance vs Marks')
plt.show()

# 3D Linear Regression Plane
study_range = np.linspace(data['StudyHours'].min(), data['StudyHours'].max(), 10)
attend_range = np.linspace(data['Attendance'].min(), data['Attendance'].max(), 10)
study_grid, attend_grid = np.meshgrid(study_range, attend_range)
marks_grid = model.predict(np.c_[study_grid.ravel(), attend_grid.ravel()]).reshape(study_grid.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['StudyHours'], data['Attendance'], data['Marks'], color='red')  # actual points
ax.plot_surface(study_grid, attend_grid, marks_grid, alpha=0.5, color='blue')  # regression plane
ax.set_xlabel('Study Hours')
ax.set_ylabel('Attendance')
ax.set_zlabel('Marks')
ax.set_title('Linear Regression Plane')
plt.show()
