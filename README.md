# Fraud Detector
Detection of fraudulent transactions (0.172% positive class) using ensemble methods and gradient boosting.

## Dataset
Here [Credit Card Fraud Detection Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) is used.

## Stack
[numpy](https://numpy.org), [pandas](https://pandas.pydata.org), [scikit-learn](https://scikit-learn.org/stable/index.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/), [optuna](https://optuna.org), [matplotlib](https://matplotlib.org)

## Key Results
- **Best model**: ExtraTreesClassifier with custom class_weight;
- **Best PR-AUC**: 0.8756;
- **Key insight**: Different algorithms require significantly different class weights (17.6 vs 193.9).

All current results are shown in `notebooks/exploration.ipynb`.

## Project Structure
- `notebooks/`: Main analysis Jupyter notebook;
- `data/`: Processed datasets;
- `results/`: Visualizations and metrics.
