# Personal-Loan-Acceptance
# Signal Desk

`Signal Desk` is a Streamlit web app that predicts whether a bank client is likely to subscribe to a term deposit campaign.

The app uses a trained PyCaret classification model and presents the prediction in a more styled, dashboard-like interface instead of a plain notebook-style layout.

## Features

- Predicts term deposit subscription outcome from client and campaign details
- Uses a saved trained model from `loan_model.pkl`
- Built with Streamlit for quick local deployment
- Clean single-result output for faster decision making

## Project Files

- `app.py` - main Streamlit application
- `loan_model.pkl` - trained machine learning model
- `bank-additional.csv` - dataset used for the project
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.9+
- pip

## Installation

```bash
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app.py
```

After running the command, Streamlit will open the app in your browser.

## How It Works

1. Enter client profile details such as age, job, marital status, education, and loan information.
2. Enter campaign and market details such as call duration, contact count, previous outcome, and macroeconomic indicators.
3. Submit the form to get the prediction result.

## Prediction Output

The app returns a simplified decision:

- Positive signal: client is likely to subscribe
- Negative signal: client is unlikely to subscribe

## Dependencies

The app currently uses:

- `streamlit`
- `pandas`
- `pycaret`

## Notes

- Make sure `loan_model.pkl` is present in the project folder before running the app.
- The UI is customized with embedded CSS inside `app.py`.
