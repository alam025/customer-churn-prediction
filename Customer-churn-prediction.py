#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Prediction
# Predicting whether a customer will leave the company or not
# Create initial version with just imports and class structure
# ## Import Libraries

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ## Data Collection and Processing

# In[2]:# Add load_data() and explore_data() methods

# Loading the CSV data to Pandas DataFrame
churn_data = pd.read_csv('customer_churn_data.csv')

# In[3]:

# Print first 5 rows of the dataset
churn_data.head()

# In[4]:

# Print last 5 rows of the dataset
churn_data.tail()

# In[5]:

# Number of rows and columns
churn_data.shape

# In[6]:

# Getting information about the dataset
churn_data.info()

# In[7]:

# Checking for missing values
churn_data.isnull().sum()

# In[8]:

# Statistical measures about the data
churn_data.describe()

# In[9]:

# Checking the distribution of Target Variable (Churn)
churn_data['Churn'].value_counts()

# ## Data Visualization

# In[10]:

# Visualize Churn Distribution
plt.figure(figsize=(6, 6))
churn_data['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
plt.title('Customer Churn Distribution')
plt.ylabel('')
plt.show()

# In[11]:

# Visualize Contract Type vs Churn
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=churn_data, palette=['#2ecc71', '#e74c3c'])
plt.title('Contract Type vs Churn')
plt.show()

# In[12]:

# Visualize Monthly Charges Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=churn_data, x='MonthlyCharges', hue='Churn', kde=True, palette=['#2ecc71', '#e74c3c'])
plt.title('Monthly Charges Distribution')
plt.show()

# ## Data Preprocessing

# In[13]:

# Handle missing values in TotalCharges column
churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
churn_data['TotalCharges'].fillna(churn_data['TotalCharges'].median(), inplace=True)

# In[14]:

# Remove customerID column (not useful for prediction)
if 'customerID' in churn_data.columns:
    churn_data = churn_data.drop('customerID', axis=1)

# In[15]:

# Encode categorical variables
label_encoder = LabelEncoder()

# List of categorical columns to encode
categorical_columns = churn_data.select_dtypes(include=['object']).columns.tolist()

# Remove 'Churn' from the list (we'll encode it separately)
if 'Churn' in categorical_columns:
    categorical_columns.remove('Churn')

# Encode all categorical columns
for column in categorical_columns:
    churn_data[column] = label_encoder.fit_transform(churn_data[column])

# In[16]:

# Encode target variable (Churn: Yes=1, No=0)
churn_data['Churn'] = churn_data['Churn'].map({'Yes': 1, 'No': 0})

# In[17]:

# Check the processed data
churn_data.head()

# ## Splitting Features and Target

# In[18]:

X = churn_data.drop(columns='Churn', axis=1)
Y = churn_data['Churn']

# In[19]:

print("Features (X):")
print(X.head())

# In[20]:

print("\nTarget (Y):")
print(Y.head())

# In[21]:

# Check the shape
print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {Y.shape}")

# ## Splitting Data into Training and Test Sets

# In[22]:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# In[23]:

print(f"Total samples: {X.shape[0]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ## Model Training
# ### Random Forest Classifier

# In[24]:

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=2)

# In[25]:

# Training the Random Forest model with training data
model.fit(X_train, Y_train)

# ## Model Evaluation

# In[26]:

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# In[27]:

print('Accuracy on Training data:', training_data_accuracy)

# In[28]:

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# In[29]:

print('Accuracy on Test data:', test_data_accuracy)

# In[30]:

# Confusion Matrix
cm = confusion_matrix(Y_test, X_test_prediction)
print("\nConfusion Matrix:")
print(cm)

# In[31]:

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# In[32]:

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))

# In[33]:

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# In[34]:

# Visualize Feature Importance
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10), color='#3498db')
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# ## Building a Predictive System

# In[35]:

# Example customer data for prediction
# (Make sure to use the same order and encoding as your training data)
input_data = X_test.iloc[0].values.reshape(1, -1)

# In[36]:

# Make prediction
prediction = model.predict(input_data)
print(f"Prediction: {prediction}")

if prediction[0] == 0:
    print("The customer will NOT churn (Stay)")
else:
    print("The customer will CHURN (Leave)")

# In[37]:

# Get prediction probability
prediction_probability = model.predict_proba(input_data)
print(f"\nPrediction Probability:")
print(f"No Churn: {prediction_probability[0][0]*100:.2f}%")
print(f"Churn: {prediction_probability[0][1]*100:.2f}%")

# In[ ]: