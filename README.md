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

## Conclusion

This project demonstrates the use of Logistic Regression for binary classification of sonar signals. The model is capable of distinguishing between signals reflected off rocks and mines with a 85% of accuracy. Further improvements can be made by experimenting with different models and feature engineering techniques.

## Requirements

- Python 3
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## How to Run

1. Clone the repository.
2. Ensure all dependencies are installed.
3. Run the script to train the model and make predictions.
        
