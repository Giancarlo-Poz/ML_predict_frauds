# Fraud Detection with Logistic Regression

A machine learning model that predicts fraudulent transactions given historical transaction data, developed as part of a data scientist take-home assessment.

## Case Study

Consider a firm that enables users to purchase content through a range of payment options known as payment channels — direct carrier billing, e-wallets, credit and debit cards, among others. Each channel is vulnerable to fraudulent activities such as stolen credit cards, phishing for one-time passwords, and unauthorised use of e-wallets.

Fraudulent activity causes revenue loss and reputational damage, so it is critical to flag suspicious transactions early. This project builds a logistic regression model that predicts whether a transaction is likely fraudulent, trained on historical transaction data that includes user-reported fraud labels.

## Dataset

The dataset contains **202,389 transactions** across 19 columns (identifiers, timestamps, geographic fields, transaction details, and a binary fraud flag).

The dataset file is not included in this repository. Download it from Google Drive and place it at the project root as `sample_dataset_data_scientist.xlsx`:

**[Download dataset from Google Drive](https://docs.google.com/spreadsheets/d/1tM5x6kcWw5hr63pdiYbPCjwlgI3NZZZi/edit?usp=drive_link&ouid=110692307547286194255&rtpof=true&sd=true)**

## Approach

### Feature Engineering

Ten features are derived from the raw columns:

| Feature | Description |
|---------|-------------|
| `TxnHour` | Hour of day the transaction completed |
| `TxnWeekday` | Day of week (0–6) |
| `TxnDuration_Seconds` | Time elapsed from initiation to completion |
| `FromFirstTxnToCurrentTxn_Days` | Days since the user's first transaction |
| `ClientIP_occurrences` | Frequency of the client IP address in the dataset |
| `User_Id_occurrences` | Frequency of the user ID |
| `Email_Id_occurrences` | Frequency of the email address |
| `GeoIpNotEqAlpha2` | Flag: GeoIP country does not match billing country |
| `FirstTransactionWithoutEmail` | Flag: email was registered after the first transaction |
| `TxnDuration_X_UniquePaymentCh` | Interaction term: transaction duration × number of unique payment channels |

### Preprocessing

- **Categorical encoding** — pandas `.cat.codes` for merchant, channel, country, device, and item fields
- **Missing values** — GeoIP nulls and missing email occurrence counts filled with 0; `FirstEmailDate` imputed from `FirstTransactionDate` where applicable
- **Outlier treatment** — Tukey's IQR method used for detection; Winsorization (1st–99th percentile capping) applied to 7 features rather than removal, since 36.9% of fraudulent transactions were statistical outliers
- **Scaling** — StandardScaler (zero mean, unit variance) applied to all features
- **Train / validation / test split** — 60% / 20% / 20%

### Model

Logistic regression (`sklearn.linear_model.LogisticRegression`) trained in three stages:

1. **Baseline** — `class_weight="balanced"` to compensate for class imbalance
2. **v1** — regularisation strength `C` and decision threshold jointly optimised via scipy's Powell method, maximising F1 score on the validation set
3. **v2** — same optimisation framework with a custom modified F1 score that weights recall more heavily (`f1_mod = 2 × precision × recall⁴ / (precision + recall⁴)`) to reduce false negatives

The decision threshold is treated as a hyperparameter and optimised alongside `C`, rather than fixed at the default 0.5.

### Evaluation

Models are evaluated on the test set using precision, recall, F1 score, and a full confusion matrix broken down as TP / FP / FN / TN rates.

## Project Structure

```
.
├── Pozzo_logistic_regression.ipynb        # Main analysis notebook
├── Pozzo_logistic_regression_slides.pdf   # Presentation slides
├── Data Scientist Take Home Exam - Instructions.pdf
└── description_of_the_dataset .pdf
```

## Requirements

- Python 3.x
- pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, openpyxl

Install dependencies:

```bash
python3 -m pip install pandas numpy scikit-learn scipy matplotlib seaborn openpyxl
```

## Usage

```bash
jupyter notebook Pozzo_logistic_regression.ipynb
```

The notebook is self-contained and runs end-to-end from raw data loading to final model evaluation.
