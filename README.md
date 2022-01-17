# customer-subscription: Outcome prediction of bank marketing calls

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Data

This project uses data from the [UCI Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing).
It refers to a dataset of direct marketing campaign calls performed by a Portuguese banking institution from 2008 to 2010.

### Source:
```
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
```

## Context

A Portuguese banking institution placed direct marketing phone calls in order to try to subscribe customers to a term deposit.
Data referring to the callee profile (age, profission, marital status, existence of credit in default, housing and personal loan), call metadata (contact form, weekday, month, call duration, number and outcome of previous calls), and the outcome of the call (whether a term deposit was subscribed) was collected.

## Goal

Evaluate the feasibility of automatically predicting the outcome of bank marketing calls based on the callee profile and call metadata.

## Development

- Add the `train_file.xlsx` and `test_file.xlsx` to the data folder.
- Install Python 3.8.12 using a virtual environment of your choice (e.g. `pyenv` or `conda`).
- Install `graphviz` using a package manager of your choice (e.g. `brew` or `conda`).
- `pip install -r dev-requirements.txt`
- `pip install -r requirements.txt`