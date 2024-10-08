# Analysis and Prediction of Bank Customer Churn
Dataset: Customer Churn Dataset.csv

Data source: https://www.kaggle.com/datasets/anandshaw2001/customer-churn-dataset

**Project Introduction**

This project aims to predict whether bank customers will churn using machine learning models. Customer churn is a significant issue in the banking industry. With accurate prediction models, banks can take measures to retain customers and reduce losses. The project is carried out in four dimensions: exploratory data analysis, customer segmentation, feature importance analysis, and churn prediction models.

**Tech Stack**

Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn

**Project Achievements**

1. Exploratory Data Analysis
+ Analyze the relationship between customer basic information, financial status, credit, behavior and preferences, and churn status through descriptive visualization and draw conclusions.
+ Discuss positive and negative correlations through the Pearson correlation coefficient and propose follow-up solutions.
+ Use KS test to determine the normal distribution in T-test/U-test, calculate with T-test for normally distributed data, and use U-test for others.
+ Use chi-square test to judge the differences between the churn and non-churn groups.
2. Customer Segmentation
+ Train with K-means clustering algorithm, identify three customer groups, and analyze the average churn rate of the three groups.
3. Feature Importance Analysis
+ Choose RandomForestClassifier as the baseline model, use GridSearchCV for model parameter tuning, and find the optimal parameter combination.
4. Model Evaluation
+ Accuracy: 0.863, indicating that the model has an accuracy of 86.3% in predicting customer churn.
+ Precision: 0.776, indicating that when the model predicts customer churn, the prediction is correct 77.6% of the time.
+ Recall: 0.425, indicating that the model can identify 42.5% of actual churned customers. This relatively low value suggests the model needs improvement in capturing all real churned customers.
+ ROC AUC: 0.866, showing the model's overall ability to distinguish between positive and negative samples, close to 0.9, indicating good classification performance.