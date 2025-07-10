# Obesity Risk Prediction Using Ensemble Learning

## Project Overview
This project implements various ensemble learning techniques to predict obesity risk categories based on health, lifestyle, and demographic features.

## Dataset
The dataset contains information about:
- Demographics (Gender, Age, Height, Weight)
- Health indicators (Family history, eating habits, physical activity)
- Lifestyle factors (Transportation, smoking, alcohol consumption)
- Target variable: Obesity categories (NObeyesdad)

## Ensemble Methods Implemented
1. **Bagging**: Random Forest
2. **Boosting**: AdaBoost, Gradient Boosting, XGBoost, LightGBM
3. **Voting**: Soft voting classifier
4. **Stacking**: Stacked generalization with logistic regression meta-learner

## Installation and Setup
1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Place your dataset file as 'dataset.csv' in the project directory
4. Run the analysis: `python obesity_prediction.py`

## Results
The project generates several output files:
- `model_results.csv`: Comparison of all models
- `exploratory_analysis.png`: Data exploration visualizations
- `model_comparison.png`: Model performance comparisons
- `confusion_matrix.png`: Confusion matrix of the best model
- `feature_importance.png`: Feature importance analysis

## Model Performance
The ensemble methods typically outperform individual baseline models by:
- Reducing overfitting through model averaging
- Combining different model strengths
- Improving generalization to unseen data

## Author
2024B3PS1042G_obesity_prediction

