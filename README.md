# Student Depression Risk Prediction  

## Overview

This project builds a supervised machine learning model to predict student depression risk using survey-based academic, behavioral, and lifestyle data.

The goal is to identify structured patterns associated with elevated mental health risk and evaluate predictive performance using classification metrics.

## Analytical Objective

- Predict likelihood of student depression risk  
- Identify key contributing behavioral factors  
- Evaluate model performance using accuracy, F1-score, and ROC-AUC  
- Demonstrate how predictive analytics could support early intervention strategies  


## Dataset Summary

The dataset contains 27,901 records, with variables including:

- Academic pressure  
- Sleep duration  
- Study hours  
- Financial stress  
- Lifestyle indicators  

Target variable:  
- Binary depression risk classification  


# Data Cleaning & Preprocessing

### Preprocessing Steps

- Checked for missing values  
- Encoded categorical variables using `LabelEncoder`  
- Applied `MinMaxScaler` to normalize numerical features  
- Performed 60/40 train-test split  

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = df.drop('target', axis=1)
y = df['target']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42
)
```

# Modeling

## Decision Tree Classifier

A Decision Tree model with balanced class weighting was used to address potential class imbalance.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)
```


# Model Performance

### Confusion Matrix

```
[[4037  806]
 [1035 3924]]
```

### Performance Metrics

- Accuracy: **81%**
- F1 Score (avg): **0.81**
- ROC-AUC: **0.81**

The model demonstrates strong class separability and balanced predictive performance across both high-risk and low-risk groups.


# Insights

- Depression risk shows measurable, structured patterns within academic and lifestyle variables.
- With 81% accuracy and AUC, the model demonstrates meaningful predictive capability suitable for early screening.
- Balanced recall across classes suggests consistent detection of both high-risk and low-risk students.
- Predictive modeling can support data-informed mental health resource prioritization while maintaining ethical safeguards.


# Tools Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  


# Limitations

- Survey-based data may contain self-report bias  
- Model performance may vary across populations  
- Ethical considerations are critical before real-world implementation  
