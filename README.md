# Human Age Prediction

This project implements a machine learning model to predict human age based on various health and lifestyle factors using a Random Forest Regressor. The dataset includes features such as blood pressure, physical activity level, smoking status, and more.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Screenshots](#screenshots)
- [License](#license)

## Project Description

The goal of this project is to predict the age of individuals based on their health metrics and lifestyle choices. The model is trained using a dataset that includes various attributes, and it employs a Random Forest Regressor for age prediction. The code performs several key tasks:

1. **Data Loading**: Reads the dataset from a CSV file.
2. **Data Preprocessing**: Cleans and transforms the data to make it suitable for modeling.
3. **Feature Engineering**: Creates new features and encodes categorical variables.
4. **Model Training**: Splits the data into training and testing sets, scales the features, and trains a Random Forest model.
5. **Hyperparameter Tuning**: Uses Randomized Search Cross-Validation to find the best model parameters.
6. **Model Evaluation**: Evaluates the trained model using metrics like Mean Absolute Error and R² Score.

## Results

The model achieved remarkable performance metrics, showcasing its effectiveness in predicting human age:

- **Mean Absolute Error (MAE)**: **3.11** 
- **R² Score**: **0.96**

These results indicate that the model's predictions are incredibly close to the actual ages, with an average error of just over **3 years**. The R² score of **0.96** suggests that the model explains **96%** of the variance in the age data, which is an outstanding achievement in predictive modeling. This level of accuracy is not only impressive but also demonstrates the model's robustness and reliability in real-world applications. Such precision can significantly aid in health assessments and personalized lifestyle recommendations.

## Screenshots

![Data Preprocessing Screenshot](Screenshot_2024-11-04_221909.png)
![Model Evaluation Output Screenshot](Screenshot_2024-11-04_221939.png)

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- scikit-learn
- numpy

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn numpy
