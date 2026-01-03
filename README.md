# Student Marks Prediction using Multiple Linear Regression

## Overview

This project implements a **Machine Learning regression model** to predict student academic performance based on **study hours** and **attendance percentage**.  
It demonstrates the complete **end-to-end ML workflow**, including data loading, preprocessing, model training, evaluation, prediction, and visualization.

The project is intended for **learning and demonstrating core Machine Learning concepts** using Python and Scikit-learn.

---

## Problem Statement

Academic performance is influenced by multiple factors.  
This project aims to model the relationship between:

- Study Hours
- Attendance Percentage

and predict the **final exam marks** using a supervised learning approach.

---

## Machine Learning Approach

- **Learning Type:** Supervised Learning  
- **Problem Type:** Regression  
- **Algorithm:** Multiple Linear Regression  

### Mathematical Model

The multiple linear regression equation is:

$$
Marks = b_0 + b_1 \times StudyHours + b_2 \times Attendance
$$

Where:  
- $b_0$ is the intercept  
- $b_1, b_2$ are coefficients learned from data  

The model estimates these parameters using the **Least Squares Optimization Method**, minimizing prediction error.

---

## Project Structure
 - Student-Marks-Prediction/
 - students.csv # Dataset containing student data
 - student_marks.py # Main Python script
 - README.md # Project documentation
 - requirements.txt # Python dependencies
 - .gitignore # Files to exclude from Git

#
### File Description
#
- **students.csv**  
  Contains the dataset columns:
  - StudyHours
  - Attendance
  - Marks

- **student_marks.py**  
  - Loads and processes the dataset  
  - Trains a Multiple Linear Regression model  
  - Evaluates model performance  
  - Predicts marks for new student inputs  
  - Generates 2D and 3D visualizations  

- **requirements.txt**  
  Lists the Python libraries required to run the project  

- **.gitignore**  
  Excludes environment files and caches like `venv/` and `__pycache__/`  

---

## Technologies and Libraries

| Tool / Library | Purpose                               |
| -------------- | ------------------------------------- |
| Python         | Core programming language             |
| Pandas         | Data loading and manipulation         |
| NumPy          | Numerical computations                |
| Matplotlib     | Data visualization (2D and 3D)       |
| Scikit-learn   | Machine Learning model and evaluation |

### Installation

```bash
pip install pandas numpy matplotlib scikit-learn
```
## Workflow Explanation

### 1. Data Loading
The dataset is read from a CSV file using **Pandas**.

### 2. Feature Selection
- **Inputs (X):** StudyHours, Attendance  
- **Output (y):** Marks

### 3. Train-Test Split
- 80% training data  
- 20% testing data

### 4. Model Training
The **Multiple Linear Regression** model is trained on the training dataset.

### 5. Prediction
- Predictions are made on the test data  
- The model also predicts marks for **user-provided inputs**

### 6. Evaluation
Performance is measured using **Mean Absolute Error (MAE)**.

---

## Model Evaluation

### Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n} \sum |Actual - Predicted|
$$

- Measures the **average prediction error**  
- Expressed in the same unit as the target variable (Marks)  
- **Lower MAE** indicates better model accuracy  

---

## Data Visualization

The project includes multiple visualizations to analyze data and model performance:

1. **Study Hours vs Marks (2D Scatter Plot)**  
   Shows correlation between study time and academic performance.

2. **Attendance vs Marks (2D Scatter Plot)**  
   Illustrates the impact of attendance on marks.

3. **Actual vs Predicted Marks**  
   Evaluates prediction accuracy using a reference diagonal line.

4. **3D Scatter Plot**  
   Displays the combined relationship between study hours, attendance, and marks.

5. **3D Regression Plane**  
   Visualizes the best-fit plane learned by the regression model.

---

## Example Usage

**Input:**
```text
Study Hours: 6
Attendance: 85
```
**Output:**
```text
Predicted Student Marks: 78.45
```
