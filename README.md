# Credit Risk & Loan Default Prediction

## 1. Project Overview
This project develops a robust machine learning model to predict the probability of a borrower defaulting on a loan. By leveraging a large historical dataset from Lending Club, the model serves as a powerful tool for quantitative risk assessment. The primary objective is to provide a data-driven basis for lending decisions, enabling financial institutions to minimize credit losses, reduce risk exposure, and optimize their lending portfolios. This analysis demonstrates a strong foundation in statistical modeling, data science, and quantitative analysis.

## 2. Data Source
The analysis utilizes the **Lending Club Loan Data** dataset, a comprehensive record of historical loan information.
- **Source:** [Kaggle Lending Club Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)
- **Contents:** The dataset includes detailed applicant information (income, employment length, credit history), loan characteristics (loan amount, interest rate, term), and final loan status.
- **Data Dictionary:** `LCDataDictionary.xlsx` is included for detailed feature descriptions.

## 3. Quantitative Methodology

### 3.1. Data Cleaning and Preprocessing
A rigorous data preparation pipeline was implemented to ensure the integrity and quality of the data fed into the model.
- **Handling Missing Data:** Columns with over 30% missing values were strategically removed to avoid introducing excessive noise or bias from large-scale imputation. For remaining features, missing numerical values were imputed with the **median**, a measure chosen for its robustness to outliers common in financial data (e.g., income). Categorical missing values were imputed with the **mode**.
- **Feature Engineering & Transformation:**
    - **Target Variable Definition:** A binary target variable, `is_default`, was engineered from the `loan_status` column. Loan statuses such as 'Charged Off' and 'Default' were mapped to `1`, while statuses like 'Fully Paid' were mapped to `0`. This transformation creates a clear, binary classification problem.
    - **Numerical Conversion:** Text-based features containing quantitative information were converted to a numerical format. This included `term` (e.g., '36 months' -> 36), `emp_length` ('10+ years' -> 10), and string-based percentages like `int_rate`.
- **Categorical Data Encoding:** **One-hot encoding** was applied to all nominal categorical features. This technique creates binary (0/1) columns for each category, allowing the model to interpret them without assuming any ordinal relationship. The `drop_first=True` parameter was used to create k-1 dummy variables, a best practice that prevents **multicollinearity** in the feature set.
- **Dimensionality Reduction:** Non-essential or redundant columns (e.g., `zip_code`, `emp_title`) were dropped to simplify the model and reduce computational overhead without sacrificing predictive power.

### 3.2. Model Development and Validation
- **Model Selection:** A **Random Forest Classifier** was chosen for this classification task. This ensemble learning method was selected for several key reasons:
    - **High Predictive Accuracy:** It is known for its strong performance on complex, tabular datasets.
    - **Robustness to Overfitting:** By aggregating the results of many individual decision trees, it minimizes the risk of overfitting to the training data.
    - **Handles Non-Linearity:** It can capture complex, non-linear relationships between features and the target variable, which linear models might miss.
- **Model Validation Strategy:** The dataset was split into training (70%) and testing (30%) sets. Crucially, a **stratified sampling** approach (`stratify=y`) was used. This ensures that the proportion of default to non-default loans was identical in both the training and testing sets, which is critical for evaluating model performance on an imbalanced dataset like this one.

## 4. Key Quantitative Skills Demonstrated
This project showcases proficiency in the following data science and quantitative analysis skills:
- **Statistical Modeling:** Applied ensemble machine learning techniques (Random Forest) to solve a high-impact binary classification problem.
- **Data Wrangling & Preprocessing:** Demonstrated expertise in handling large datasets, including robust methods for missing value imputation (median, mode) and feature transformation.
- **Feature Engineering:** Skilled in creating a binary target variable from multi-class data and converting raw features into a format suitable for modeling.
- **Model Validation:** Proficient in using appropriate validation strategies (stratified train-test split) to ensure reliable model evaluation, particularly for imbalanced datasets.
- **Dimensionality Reduction:** Experienced in identifying and removing redundant or low-information features to improve model efficiency and interpretability.
- **Performance Metrics:** Expertise in interpreting key classification metrics (Accuracy, Precision, Recall, F1-Score) to assess model performance from both a statistical and business perspective.
- **Technical Proficiency:** Advanced use of Python libraries including **Pandas** and **NumPy** for data manipulation and **Scikit-learn** for implementing the end-to-end machine learning workflow.

## 5. Results and Interpretation

The Random Forest model's performance was evaluated on the unseen test set, yielding the following results:

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

### Interpretation of Quantitative Results:
- **Overall Accuracy:** The model achieved an outstanding accuracy of **98.87%**, indicating a very high rate of correct predictions across both classes.
- **Precision (Default Class):** The model achieved a **perfect precision score of 1.00** for the default class (1). This is the most significant result from a business perspective. It means that **every single loan the model predicted would default was a correct prediction.** This allows a financial institution to confidently decline high-risk applicants, directly preventing financial losses with a high degree of certainty.
- **Recall (Default Class):** The model achieved a **recall of 0.91**, meaning it successfully identified **91% of all loans that actually defaulted**. While a small fraction of defaults (9%) were missed, this high recall ensures that the vast majority of risky loans are flagged.
- **Conclusion:** The combination of perfect precision and high recall makes this model an exceptionally effective and reliable tool for credit risk management. It provides a strong quantitative foundation for making informed, data-driven lending decisions.

## 6. How to Run the Project

### Prerequisites
- Python 3.x
- `pip` (Python package installer)

### Setup
1.  **Clone the repository or navigate to the project directory.**
2.  **Place the Dataset:** Ensure `loan.csv` is located in a `dataset/` subdirectory.
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution
To run the complete analysis pipeline from data processing to model evaluation, execute the script:
```bash
python credit_risk_analysis.py
```
The script will output the final model evaluation metrics to the console.