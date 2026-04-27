# Lab 4: Regression and Classification

Introduction to Artificial Intelligence  
Warsaw University of Technology, Summer 2026  
Group 16  
Authors: Anatoli Abahumna and Leonardo Meloni

## Project

This project predicts house prices using the California Housing dataset from
scikit-learn.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

The script loads the dataset, performs feature analysis, creates train/test
CSV files, trains Ridge Regression and Random Forest models, runs 4-fold
cross-validation, and saves the model results.

## Note About Large Files

The trained Random Forest model file is not included because it is large. It will be generated automatically when running:

```bash
python3 main.py
