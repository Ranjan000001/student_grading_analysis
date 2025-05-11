# 🎓 Student Grading Analysis

## 📘 Project Overview

The project involves analyzing students grading data from including midterm scores, Participation Score, Projects_Score and avcerage score of  Assignments, Quizzes ,  Gender , Department , Extracurricular Activities, Internet Access at Home, Parent Education Level, Family_Income_Level, Grade. The analysis encompasses data cleaning, featur engineering and visualization to understand student grading patterns and trends.

## Data Cleaning and Preprocessing

Import Libraries 
```jupyter
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
```
Read files
```jupyter
# store csv data
df = pd.read_csv(r"C:\Users\user\Downloads\Students_Grading_Dataset_Biased.csv")  # biased data

student_df = pd.read_csv(r"C:\Users\user\Downloads\Students_Grading_Dataset.csv")   # original data
```
Information of data
```jupyter
# info and head
student_df.info()
student_df.head()
```
Cherck null cells
```jupyter
# checking null cells
student_df.isnull().sum()
```
Columns name
```jupyter
# column 
student_df.columns
```
## EDA

### Univariate
### Numerical
```jupyter
# describe of columns
columns = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score','Projects_Score', 'Total_Score','Sleep_Hours_per_Night','Study_Hours_per_Week', 'Stress_Level (1-10)']
for col in columns:
    print(student_df[col].describe())
```

```jupyter
columns = ['Age', 'Attendance (%)','Midterm_Score','Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score','Projects_Score', 'Total_Score','Sleep_Hours_per_Night','Study_Hours_per_Week', 'Stress_Level (1-10)']
for col in columns:
    student_df[col].plot(kind='kde')
    plt.show()
```
### Categoreials
```jupyter
cat_columns = ['Student_ID', 'First_Name', 'Last_Name', 'Email', 'Gender','Department','Extracurricular_Activities', 'Internet_Access_at_Home','Parent_Education_Level', 'Family_Income_Level','Grade']
for col in cat_columns:
    print(student_df[col].value_counts())
```


```jupyter
cat_columns = [ 'First_Name', 'Last_Name', 'Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home','Parent_Education_Level', 'Family_Income_Level','Grade']
for col in cat_columns:
    count = student_df[col].value_counts()
    count.plot(kind = 'pie', autopct='%1.1f%%',figsize=(6,6))
    plt.title(f'{col}')
    plt.axis('equal')
    plt.legend()
    plt.show()
```


```jupyter
cat_columns = ['First_Name', 'Last_Name', 'Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home','Parent_Education_Level', 'Family_Income_Level','Grade']
for col in cat_columns:
    count = student_df[col].value_counts()
    count.plot(kind='bar')
    plt.show()
```


```jupyter
# check the total score
total = student_df[['Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 'Projects_Score']].mean(axis=1)  # calculate average of all scores
print((total==student_df['Total_Score']).all())  # check all values of total and total score are same or note
```
### Multivariate

```jupyter
new_df =student_df.copy()
new_df['First_Name'] = new_df['First_Name'].map({'Omar':0, 'Maria':1, 'Ahmed':2, 'John':3, 'Liam':4, 'Sara':5, 'Emma':6, 'Ali':7})
new_df['Last_Name'] = new_df['Last_Name'].map({'Williams':0, 'Brown':1, 'Jones':2, 'Smith':3, 'Davis':4, 'Johnson':5})
new_df['Department'] = new_df['Department'].map({'Mathematics':0, 'Business':1, 'Engineering':2, 'CS':3})
new_df['Extracurricular_Activities'] = new_df['Extracurricular_Activities'].map({'Yes':0, 'No':1})
new_df['Internet_Access_at_Home'] = new_df['Internet_Access_at_Home'].map({'Yes':0, 'No':1})
new_df['Grade'] = new_df['Grade'].map({'A':0, 'B':1, 'C':2, 'D':3, 'F':4})
new_df['Gender'] = new_df['Gender'].map({'Male':0, 'Female':1})
```

```jupyter
# correlation matrix
corr_matrix = new_df.select_dtypes(include=np.number).corr()
# heat map
plt.figure(figsize=(12, 7))
sns.heatmap(corr_matrix, annot = True, fmt ='.2f',cmap='coolwarm', center=0, linewidths = 0.5)
plt.title('Heatmap of corelation matrix')
plt.show()
```


```jupyter
cat_columns = [ 'First_Name', 'Last_Name', 'Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home','Parent_Education_Level', 'Family_Income_Level','Grade']
for col in cat_columns:
    count = student_df[col].value_counts()
    count.plot(kind = 'pie', autopct='%1.1f%%',figsize=(6,6))
    plt.title(f'{col}')
    plt.axis('equal')
    plt.legend()
    plt.show()
```

### Feature Engineering
```jupyter
# Feacture Engineering
student_df['Full_name'] = student_df['First_Name'] + student_df['Last_Name']
```
## ML Model basics
### Adding missing values
```jupyter
features = ['Midterm_Score', 'Final_Score', 'Assignments_Avg',
            'Quizzes_Avg', 'Participation_Score', 'Projects_Score']

# Now drop rows with missing values in features or Total_Score
df_cleaned = student_df.dropna(subset=features + ['Total_Score'])

# Split into X (features) and y (target)
X = df_cleaned[features]
y = df_cleaned['Total_Score']
print(X.isnull().sum())
print(y.isnull().sum())
```

### Reverse Engerning the formula
```jupyter

model = LinearRegression()
model.fit(X, y)

# Coefficients
weights = pd.Series(model.coef_, index=features)
intercept = model.intercept_
print("Weights:\n", weights)
print("Intercept:", intercept)
print("R² score:", model.score(X, y))
```

### Trying Tree Model
```jupyter
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X, y)
print("Random Forest R²:", model_rf.score(X, y))
```

### Feacher Introduction
```jupyter
# Assuming model_rf is trained already on X_filled and y
importances = model_rf.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feat_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot it
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df)
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.show()
```
### Cross validation
```jupyter
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Re-initialize your Random Forest with current parameters
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')

print("Cross-validated R² scores:", cv_scores)
print("Average R² score:", np.mean(cv_scores))
```

### KNN Impact
```jupyter
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

### Retain and Cross Validation Again
```jupyter
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_model, X_imputed, y, cv=5, scoring='r2')

print("Cross-validated R² scores after KNN Imputation:", cv_scores)
print("Average R² score:", np.mean(cv_scores))
```

### Checking Target distribution
```jupyter
sns.histplot(y, kde=True)
```

### Try Similar Model
```jupyter
from sklearn.linear_model import RidgeCV, LassoCV
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10])
lasso = LassoCV(cv=5)

ridge.fit(X_imputed, y)
lasso.fit(X_imputed, y)

print("Ridge R²:", ridge.score(X_imputed, y))
print("Lasso R²:", lasso.score(X_imputed, y))
```

## 📈 Key Insights
- Gender-based performance variations.

- Influence of parental education on student scores.

- Subject score correlations.
