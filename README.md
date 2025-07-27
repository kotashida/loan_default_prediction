# Credit Risk Default Prediction

## Project Overview
This project aims to develop a machine learning model to predict loan defaults based on a comprehensive dataset of applicant information and credit history. The primary goal is to identify key risk indicators and build a predictive model that can assess the likelihood of a loan applicant defaulting. This can help financial institutions make more informed lending decisions, mitigate risks, and optimize their loan portfolios.

## Data Source
The dataset used in this project is the **Lending Club Loan Data**, obtained from Kaggle: [https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv). It contains historical loan data, including applicant information, loan characteristics, and loan status.

## Key Steps and Methodology

### 1. Data Acquisition
The dataset (`loan.csv`) was manually downloaded from Kaggle and extracted into the `dataset/` directory. The data dictionary (`LCDataDictionary.xlsx`) is also included for reference.

### 2. Data Cleaning and Preprocessing
This phase involved several critical steps to prepare the raw data for machine learning:
- **Handling Missing Values:** Columns with more than 30% missing values (e.g., `id`, `member_id`, hardship-related features) were dropped. Remaining missing numerical values were imputed with the median, and categorical values with the mode.
- **Feature Selection/Engineering:**
    - Columns with only one unique value (constant columns) were removed.
    - Non-essential or highly granular columns like `zip_code`, `issue_d`, `earliest_cr_line`, `title`, `emp_title`, `last_pymnt_d`, and `last_credit_pull_d` were dropped to reduce dimensionality and noise. (Note: In a more advanced project, date features would be engineered).
    - `term` (e.g., '36 months') and `emp_length` (e.g., '10+ years') were converted to numerical representations.
    - `int_rate` and `revol_util` were converted from string percentages to float numbers.
- **Categorical Encoding:** All remaining categorical features were converted into numerical format using one-hot encoding (`pd.get_dummies`), with `drop_first=True` to avoid multicollinearity.
- **Target Variable Definition:** The `loan_status` column was transformed into a binary target variable `is_default` (1 for default, 0 for non-default). Loan statuses like 'Charged Off', 'Default', 'Late (31-120 days)', and 'Does not meet the credit policy. Status:Charged Off' were classified as default.

### 3. Model Building and Evaluation
- **Data Splitting:** The preprocessed dataset was split into training (70%) and testing (30%) sets to evaluate the model's generalization performance.
- **Model Selection:** A **RandomForestClassifier** was chosen for its robustness, ability to handle complex relationships, and good performance on tabular data.
- **Training:** The model was trained on the training data.
- **Evaluation:** The model's performance was assessed on the unseen test data using standard classification metrics:
    - **Accuracy:** The proportion of correctly classified instances.
    - **Precision:** The proportion of true positive predictions among all positive predictions.
    - **Recall:** The proportion of true positive predictions among all actual positive instances.
    - **F1-Score:** The harmonic mean of precision and recall.

## Project Results
The trained RandomForestClassifier model achieved the following performance on the test set:

```
Model Evaluation:
Accuracy: 0.9887
Classification Report:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99    592898
           1       1.00      0.91      0.95     85303

    accuracy                           0.99    678201
   macro avg       0.99      0.96      0.97    678201
weighted avg       0.99      0.99      0.99    678201
```

**Interpretation of Results:**
- The model demonstrates a high overall accuracy of **98.87%**, indicating its strong ability to correctly classify loan applications.
- For the non-default class (0), the model shows excellent precision, recall, and F1-score (all 0.99 or 1.00), meaning it is very good at identifying non-defaulting loans and rarely misclassifies them.
- For the default class (1), the model achieves a precision of **1.00** and a recall of **0.91**, resulting in an F1-score of **0.95**. This is a very strong result, especially the perfect precision, which means that when the model predicts a loan will default, it is almost always correct. A recall of 0.91 indicates that it identifies 91% of actual defaulting loans.

These results suggest that the model is highly effective in predicting loan defaults, which is crucial for risk management in financial lending.

## How to Run the Project

### Prerequisites
- Python 3.x
- `pip` (Python package installer)

### Setup
1.  **Clone the repository (if applicable) or navigate to the project directory:**
    ```bash
    cd credit_risk_default_prediction
    ```

2.  **Ensure the dataset is in place:**
    Make sure `loan.csv` and `LCDataDictionary.xlsx` are located in the `dataset/` subdirectory. If not, download `archive.zip` from the Kaggle Lending Club Loan Data page and extract its contents into a `dataset/` folder within the project root.

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution
To run the data processing, model training, and evaluation, execute the main Python script:

```bash
python credit_risk_analysis.py
```

The script will print the model's evaluation metrics to the console upon completion.

## File Structure
```
credit_risk_default_prediction/
├── dataset/
│   ├── loan.csv
│   └── LCDataDictionary.xlsx
├── credit_risk_analysis.py
├── requirements.txt
└── README.md
```
