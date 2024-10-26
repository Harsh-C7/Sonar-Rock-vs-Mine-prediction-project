# Sonar Mine vs Rock Prediction

This project involves building a machine learning model to classify sonar signals as either mines or rocks. The dataset used for this project is the Sonar dataset, which contains 60 features representing sonar signal returns from various angles. The goal is to accurately predict whether a given sonar signal is reflected off a rock or a mine.

## Project Overview

The project utilizes a Logistic Regression model to perform binary classification on the Sonar dataset. The dataset is split into training and testing sets to evaluate the model's performance. The model is trained using the training set and evaluated on both the training and testing sets to determine its accuracy.

## Dataset

The dataset used in this project is the Sonar dataset, which can be found in kaggle. It contains 208 samples with 60 features each, representing the intensity of sonar signals at different angles. The 61st column is the target variable, indicating whether the signal is from a rock (R) or a mine (M).

## Project Structure

- **Data Loading**: The dataset is loaded using [Pandas](https://pandas.pydata.org/) and a brief exploratory analysis is performed to understand its structure.
- **Data Preprocessing**: The features and target variable are separated. The dataset is split into training and testing sets using an 90-10 split.
- **Model Training**: A Logistic Regression model is trained on the training data.
- **Model Evaluation**: The model's accuracy is evaluated on both the training and testing sets.
- **Prediction**: The model is used to predict the class of a new sonar signal.

## Code Explanation

1. **Import Libraries**:
   ```python
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   ```

2. **Load and Explore Data**:
   ```python
   df = pd.read_csv("/content/sonar data.csv", header=None)
   df.sample(5)
   df.shape
   ```

3. **Data Preprocessing**:
   - Features and target variable are separated.
   - Data is split into training and testing sets.
   ```python
   x = df.drop(columns=60, axis=1)
   y = df[60]
   x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.1)
   ```

4. **Model Training**:
   ```python
   lr = LogisticRegression()
   lr.fit(x_train, y_train)
   ```

5. **Model Evaluation**:
   - Evaluate accuracy on training data.
   - Evaluate accuracy on testing data.
   ```python
   x_train_prediction = lr.predict(x_train)
   accuracy = accuracy_score(x_train_prediction, y_train)
   print("Training Accuracy:", accuracy)

   x_test_prediction = lr.predict(x_test)
   accuracy = accuracy_score(x_test_prediction, y_test)
   print("Testing Accuracy:", accuracy)
   ```

6. **Prediction on New Data**:
   - Predict the class of a new sonar signal.
   ```python
   input_data = (0.0307, 0.0523, 0.0653, 0.0521, 0.0611, 0.0577, 0.0665, 0.0664, 0.1460, 0.2792, 0.3877, 0.4992, 0.4981, 0.4972, 0.5607, 0.7339, 0.8230, 0.9173, 0.9975, 0.9911, 0.8240, 0.6498, 0.5980, 0.4862, 0.3150, 0.1543, 0.0989, 0.0284, 0.1008, 0.2636, 0.2694, 0.2930, 0.2925, 0.3998, 0.3660, 0.3172, 0.4609, 0.4374, 0.1820, 0.3376, 0.6202, 0.4448, 0.1863, 0.1420, 0.0589, 0.0576, 0.0672, 0.0269, 0.0245, 0.0190, 0.0063, 0.0321, 0.0189, 0.0137, 0.0277, 0.0152, 0.0052, 0.0121, 0.0124, 0.0055)
   input_data_to_numpy_array = np.array([input_data])
   predicted_result = lr.predict(input_data_to_numpy_array)
   print("Predicted Result:", predicted_result)
   ```

## Conclusion

This project demonstrates the use of Logistic Regression for binary classification of sonar signals. The model is capable of distinguishing between signals reflected off rocks and mines with a 85% of accuracy. Further improvements can be made by experimenting with different models and feature engineering techniques.

## Requirements

- Python 3.x
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## How to Run

1. Clone the repository.
2. Ensure all dependencies are installed.
3. Run the script to train the model and make predictions.
        
