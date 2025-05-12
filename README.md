# WeatherNueralNetwork

This project applies a Neural Network in order to predict the weather given a multitude of parameters. The goal of this project is to compare how a neural network compares the Mean-Squared-Error
of this model to the Mean-Squared Error of both Random Forest Classifier and Bagging Classifier (SVC).

## Dataset

- https://www.kaggle.com/datasets/nikhil7280/weather-type-classification
- Name : Weather Type Classification
- Author : Nikhil Narayan
- All features were utilized
  - Label Variable was `Weather Type` 

## File Overview

- `Output.py` : Model Training and Evaluation.
- `Weather Cleaning.py` : File cleans and creates the train and validation sets for training.
- `weater_classification_data.csv` : Full Dataset from Kaggle.
- `train_set.csv` : Train set for Model.
- `validate_set.csv` : Validation set for Model.
- `README.md` : Your reading it right now!

## Model Overview

- Optimizer : Adaptive Moment Estimation
- Convolution : Linear Transformations
  - Rectified Linear unit (ReLU)
    - 10 to 5
  - 5 to 1
- Loss Function
  - Mean Squared Error

## Results

|         Model          | Mean Squared Error |
|------------------------|--------------------|
|Random Forest Classifier|        0.372       |
|Bagging Classifier      |        0.518       |
|Neural Network          |        0.778       |

  The Neural Network in this case struggles achieve the same results of Bagging Classifier and Random Forest Classifier, this is most likely due to RFC or Bagging being better suited
the dataset. RFC in particular has a MSE of 0.372, which is very good for this dataset.


