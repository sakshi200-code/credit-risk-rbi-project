# Credit Risk Modelling — Retail Lending Portfolio

### Credit Risk Analytics | Predictive Modelling | Logistic Regression | Risk Segmentation | Stress Testing

> An end-to-end credit risk analytics case study that estimates retail loan default risk, identifies key risk drivers, validates model discrimination, converts predicted probabilities into borrower risk bands, and stress-tests the portfolio under adverse affordability and interest-rate scenarios.

---

##  Project Overview

Credit risk modelling is a core part of retail banking and lending. Financial institutions need to estimate the probability that a borrower will default so that they can make better decisions around underwriting, pricing, portfolio monitoring, and risk management.

In this project, I built an end-to-end **credit default risk modelling workflow** using a retail lending portfolio containing **32,500+ records**, with an observed default rate of approximately **21.8%**.

The project goes beyond simply training a machine-learning model.

I followed a credit-risk analyst workflow covering:

* Data quality auditing
* Outlier and unrealistic-value treatment
* Segment-level default-rate analysis
* Business-oriented feature engineering
* Logistic regression modelling
* Model validation using ROC-AUC and KS statistic
* Risk-band construction
* Model-driver interpretation
* Portfolio stress testing
* Automated reporting

The final model achieved a **ROC-AUC of 0.86** and a **KS statistic of 0.59**, demonstrating strong discriminatory power between higher-risk and lower-risk borrowers.

The predicted probabilities were also converted into **five risk bands**, where observed default rates increased from **2.5% in the lowest-risk band to 68.9% in the highest-risk band**.

---

#  Business Objective

The primary objective is to estimate the probability of credit default for retail loan applicants using borrower, loan, and credit-history characteristics.

The analysis is designed around practical banking questions:

* Which borrower segments have higher default rates?
* Which loan characteristics are associated with higher credit risk?
* How does loan grade relate to observed default?
* How does loan affordability relate to default?
* Can borrower risk be ranked effectively?
* Can predicted probabilities be converted into meaningful risk bands?
* Which variables are the strongest drivers of modelled risk?
* How might portfolio risk change under adverse interest-rate and affordability scenarios?
* How can the model support underwriting and portfolio-monitoring decisions?

---

# Business Context

The project is framed around a **retail lending portfolio**.

A credit-risk team typically needs to answer two related questions:

### 1. Individual borrower risk

> How likely is this borrower to default?

### 2. Portfolio risk

> How does the overall risk profile change when borrower or economic conditions deteriorate?

This project addresses both perspectives.

The modelling layer estimates borrower-level risk, while the risk-band and stress-testing layers translate model outputs into portfolio-level insights.

---

#  End-to-End Workflow

```text
Raw Lending Dataset
        ↓
Data Quality Audit
        ↓
Outlier & Invalid-Value Treatment
        ↓
Segment-Level Default Analysis
        ↓
Feature Engineering
        ↓
Logistic Regression
        ↓
Model Validation
   ├── ROC-AUC
   ├── KS Statistic
   └── Classification Metrics
        ↓
Risk Probability
        ↓
Five Risk Bands
        ↓
Model Driver Interpretation
        ↓
Portfolio Stress Testing
        ↓
Business Recommendations
        ↓
Automated Reports
```

---

#  Dataset Overview

The project uses a retail credit-risk dataset containing **32,500+ loan records**.

### Key data areas

The dataset contains information relating to:

###  Borrower Characteristics

* Age
* Income
* Employment length
* Home ownership
* Credit history

###  Loan Characteristics

* Loan amount
* Loan interest rate
* Loan purpose
* Loan grade
* Loan amount relative to income

###  Credit History

* Previous default history
* Length of credit history

###  Target

The target variable represents whether the loan resulted in a default.

The portfolio has an observed default rate of approximately:

```text
21.8%
```

---

#  1. Data Quality Audit

Before modelling, I performed a structured data-quality review.

The audit examines:

* Data types
* Missing values
* Non-null counts
* Cardinality
* Numeric distributions
* Categorical distributions
* Potentially unrealistic observations

The purpose was to avoid allowing data-quality problems to influence model estimates.

---

#  2. Data Cleaning & Outlier Treatment

Credit-risk models are particularly sensitive to unrealistic borrower and financial values.

The cleaning process addressed several business-invalid observations.

### Borrower age

Records with borrower ages outside a reasonable range of:

```text
18–100 years
```

were removed.

### Employment length

Unrealistic employment lengths above:

```text
60 years
```

were removed.

### Income

Records with non-positive income were excluded.

### Loan amount

Records with non-positive loan amounts were excluded.

### Missing numerical values

Missing numerical values are handled within the modelling pipeline using imputation.

This approach keeps the modelling workflow reproducible and avoids manually altering the raw dataset.

---

#  3. Segment-Level Default Rate Analysis

Before building the model, I analyzed where default risk is concentrated across the portfolio.

A model score by itself is not enough for a business discussion.

A credit-risk analyst also needs to understand:

> **Which segments are actually riskier?**

I therefore calculated default rates across important categorical and numerical segments.

### Categorical segments

Default rates were analyzed by:

* Home ownership
* Loan intent
* Loan grade
* Previous default history

### Numerical segments

Numerical variables were divided into quantile-based bands to compare default rates across borrower groups.

The analysis includes:

* Age
* Income
* Employment length
* Loan amount
* Interest rate
* Loan percentage of income
* Credit-history length

This provides a portfolio-level view of where risk is concentrated before the predictive model is introduced.

---

#  4. Feature Engineering

I created a small set of **business-interpretable features** rather than relying on highly complex transformations.

The goal was to create variables that a credit-risk analyst or underwriter could understand and explain.

### Loan-to-Income Ratio

```text
Loan-to-Income Ratio =
Loan Amount / Annual Income
```

This captures the relative size of the borrowing request compared with the borrower's income.

---

### High Interest Rate Flag

Borrowers were flagged based on whether their interest rate was at or above the portfolio median.

```text
high_interest_flag
```

This provides a simple way to represent relatively higher borrowing costs.

---

### Short Credit History Flag

Borrowers with a credit history of three years or less were flagged:

```text
short_credit_history_flag
```

A shorter credit history provides less historical evidence about borrower behaviour.

---

### Prior Default Flag

Previous default information was converted into a binary indicator:

```text
prior_default_flag
```

This makes previous default history directly usable as a model feature.

---

#  5. Model Selection — Logistic Regression

I selected **Logistic Regression** as the primary modelling approach.

The objective was not simply to maximize predictive complexity.

In credit-risk applications, interpretability is extremely important.

A model may need to be explained to:

* Underwriters
* Credit-risk teams
* Risk committees
* Management
* Regulators

Logistic regression provides:

* Interpretable coefficients
* Transparent risk direction
* Stable classification behaviour
* Straightforward probability estimates
* Easier business explanation

This makes it a strong baseline model for a credit-risk case study.

---

#  6. Model Validation

The model was evaluated using several classification and discriminatory-power metrics.

### Primary metrics

* ROC-AUC
* KS Statistic
* Accuracy
* Precision
* Recall
* F1 Score

---

##  ROC-AUC

The model achieved:

```text
ROC-AUC = 0.86
```

This indicates strong ability to distinguish between defaulting and non-defaulting borrowers.

---

##  KS Statistic

The model achieved:

```text
KS = 0.59
```

The KS statistic is particularly relevant in credit-risk modelling because it measures the separation between the distributions of good and bad borrowers.

The combination of:

```text
AUC = 0.86
KS  = 0.59
```

indicates strong risk-ranking performance for this case study.

---

#  7. Risk Band Construction

Rather than presenting only individual predicted probabilities, I converted model outputs into **five borrower risk bands**.

This makes the model much easier to use from a portfolio-management perspective.

The risk bands allow borrowers to be grouped from lower risk to higher risk.

### Observed default-rate pattern

```text
Lowest Risk Band       → 2.5% default rate
        ↓
        ↓
        ↓
Highest Risk Band      → 68.9% default rate
```

The default rate increases monotonically across the risk bands.

This is an important result because it demonstrates that the model is not merely producing probabilities—it is successfully **rank-ordering borrowers by observed risk**.

---

#  Why Risk Bands Matter

A banking team typically doesn't want to make decisions based only on:

```text
Probability = 0.43782
```

Instead, it is easier to communicate:

```text
Low Risk
Moderate Risk
Medium Risk
High Risk
Very High Risk
```

Risk bands can support differentiated:

* Underwriting review
* Approval thresholds
* Pricing
* Collateral requirements
* Monitoring intensity
* Portfolio reporting

---

#  8. Model Driver Interpretation

One of the advantages of logistic regression is that model coefficients can be interpreted.

The project extracts the strongest model drivers and evaluates their coefficient direction.

### Interpretation principle

```text
Positive coefficient
        ↓
Higher estimated default risk

Negative coefficient
        ↓
Lower estimated default risk
```

The most important model drivers include variables related to:

* Income
* Loan grade
* Interest rate
* Credit history length
* Loan affordability

These drivers should still be reviewed with business judgement before being used in a production decision system.

---

#  9. Portfolio Stress Testing

A major part of the project is the **portfolio stress-testing framework**.

Instead of asking only:

> "How does the model perform under historical conditions?"

I also asked:

> "What happens if borrower affordability and borrowing costs deteriorate?"

The stress-testing framework modifies:

* Loan interest rate
* Loan percentage of income

to simulate adverse conditions.

---

##  Stress Scenario Results

The stress scenarios indicate that average predicted default risk can increase by approximately:

```text
27% → 69%
```

from milder to more severe adverse scenarios.

This demonstrates why stress testing is important for portfolio risk management.

A model can perform well under normal conditions while the portfolio may still become significantly riskier under adverse assumptions.

---

#  10. Business Recommendations

Based on the analysis, I would recommend the following actions.

### 1. Use risk bands for differentiated underwriting

Rather than using a single pass/fail threshold, use the five risk bands to differentiate underwriting treatment.

Possible applications include:

* Approval thresholds
* Pricing
* Collateral requirements
* Manual review

---

### 2. Prioritize the highest-risk segments

The top two risk bands have substantially higher observed default rates than the portfolio average.

These borrowers should receive enhanced review and monitoring.

---

### 3. Incorporate stress testing into portfolio planning

The stress-testing results show that adverse interest-rate and affordability conditions can materially increase predicted default risk.

Stress scenarios can therefore support:

* Provisioning discussions
* Capital planning
* Risk appetite discussions
* Portfolio monitoring

---

### 4. Monitor model drivers over time

Model coefficients are based on historical relationships.

Those relationships may change as:

* Borrower behaviour changes
* Lending policies change
* Interest rates change
* Economic conditions change

Therefore, model drivers should be periodically reviewed.

---

### 5. Strengthen validation before production use

Before production deployment, the model should undergo:

* Out-of-time validation
* Calibration analysis
* Fairness testing
* Population Stability Index monitoring
* Challenger-model comparison

---

#  RBI / Banking Interview Perspective

This project was designed to demonstrate the **credit-risk lifecycle** rather than only machine-learning implementation.

The workflow can be summarized as:

```text
Define Default
      ↓
Audit Data
      ↓
Clean Data
      ↓
Analyze Default Rates
      ↓
Engineer Risk Drivers
      ↓
Build Model
      ↓
Measure AUC & KS
      ↓
Create Risk Bands
      ↓
Interpret Drivers
      ↓
Stress Portfolio
      ↓
Recommend Risk Actions
```

This is the framework I would use to explain the project during a credit-risk or banking interview.

---

#  Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression

### Reporting

* Markdown
* CSV
* Automated report generation

### Development Environment

* Jupyter Notebook
* Google Colab
* GitHub

---

#  Project Architecture

The project is structured using a modular Python implementation rather than keeping all logic inside a single notebook.

```text
credit-risk-rbi-project/
│
├──  credit_risk_case_study.ipynb
│
├──  src/
│   ├── data_loading.py
│   ├── model_pipeline.py
│   ├── reporting.py
│   └── stress_testing.py
│
├──  data/
│   ├── archive.zip
│   └── credit_risk_dataset.csv
│
├──  reports/
│   ├── model_summary.md
│   ├── interview_notes.md
│   ├── full_case_study.md
│   └── tables/
│
└──  README.md
```

---

#  Modular Components

### `data_loading.py`

Responsible for:

* Loading the dataset
* Extracting the required data
* Data-quality checks
* Cleaning invalid records

### `model_pipeline.py`

Responsible for:

* Feature preparation
* Model training
* Model evaluation
* Probability generation
* Risk-band creation
* Model-driver extraction

### `stress_testing.py`

Responsible for:

* Creating adverse scenarios
* Applying stress assumptions
* Generating stressed probability-of-default estimates

### `reporting.py`

Responsible for:

* Model summary generation
* Interview notes
* Report creation

---

#  Generated Outputs

The project automatically generates:

### Model Summary

```text
reports/model_summary.md
```

Contains the major modelling and stress-testing results.

### Interview Notes

```text
reports/interview_notes.md
```

Provides a concise way to discuss the project during an interview.

### Full Case Study

```text
reports/full_case_study.md
```

Combines:

* Business context
* Data audit
* Segment analysis
* Model results
* Risk bands
* Model drivers
* Stress testing
* Limitations

### Analytical Tables

```text
reports/tables/
```

Contains exported CSV summaries used throughout the analysis.

---

#  How to Run the Project

## 1. Clone the repository

```bash
git clone https://github.com/sakshi200-code/credit-risk-rbi-project.git
```

## 2. Navigate to the repository

```bash
cd credit-risk-rbi-project
```

## 3. Install dependencies

```bash
pip install pandas numpy scikit-learn
```

## 4. Place the dataset

The project expects the dataset archive in:

```text
data/archive.zip
```

The extracted dataset is expected at:

```text
data/credit_risk_dataset.csv
```

## 5. Run the project

```bash
python run_project.py
```

The pipeline performs:

```text
Data Loading
    ↓
Data Audit
    ↓
EDA Tables
    ↓
Model Training
    ↓
Model Validation
    ↓
Stress Testing
    ↓
Report Generation
```

---

#  Key Project Results

| Metric                                      |              Result |
| ------------------------------------------- | ------------------: |
| Portfolio Records                           |             32,500+ |
| Observed Default Rate                       |               21.8% |
| Model                                       | Logistic Regression |
| ROC-AUC                                     |            **0.86** |
| KS Statistic                                |            **0.59** |
| Risk Bands                                  |               **5** |
| Lowest-Risk Band Default Rate               |            **2.5%** |
| Highest-Risk Band Default Rate              |           **68.9%** |
| Stress-Test Increase in Avg. Predicted Risk |        **~27%–69%** |

---

#  Key Takeaways

### 1. The model has strong discriminatory power

An AUC of **0.86** and KS of **0.59** indicate that the model can effectively distinguish higher-risk borrowers from lower-risk borrowers.

### 2. Risk bands provide business value

The increase in observed default rate from **2.5% to 68.9%** across the five bands demonstrates clear risk ranking.

### 3. Interpretability matters

Logistic regression provides transparent model drivers, making it easier to explain risk decisions.

### 4. Stress testing adds a portfolio perspective

The project moves beyond individual borrower prediction and evaluates how risk could change under adverse affordability and interest-rate conditions.

### 5. Credit-risk modelling is more than model accuracy

A useful credit-risk framework combines:

```text
Data Quality
+
Risk Segmentation
+
Predictive Modelling
+
Model Validation
+
Risk Ranking
+
Stress Testing
+
Business Interpretation
```

---

#  Limitations

This project is an analytical and educational case study and should not be treated as a production credit-decision system.

### 1. No macroeconomic time series

The dataset does not contain historical macroeconomic time-series variables.

### 2. No out-of-time validation

There is no date variable suitable for a proper temporal validation framework.

### 3. Calibration has not been fully evaluated

Future work should include calibration curves and probability calibration.

### 4. Fairness analysis is not included

A production credit model should be assessed for potential fairness and disparate-impact concerns.

### 5. Population Stability Monitoring

PSI or similar monitoring should be incorporated before production deployment.

### 6. Challenger models

Alternative models could be developed and compared against the logistic-regression baseline.

---

#  Future Improvements

The next iteration of this project could include:

###  Out-of-Time Validation

Use a temporal split to evaluate whether model performance remains stable on future observations.

###  Probability Calibration

Evaluate whether predicted probabilities accurately represent observed default frequencies.

###  Fairness Analysis

Assess model performance and outcomes across relevant borrower groups.

###  Population Stability Index

Monitor whether the borrower population changes over time.

###  Challenger Models

Compare logistic regression with:

* Decision Trees
* Random Forest
* Gradient Boosting
* XGBoost

while retaining logistic regression as an interpretable benchmark.

###  Portfolio Monitoring Dashboard

Build a Power BI dashboard containing:

* Default rate
* Risk-band distribution
* AUC / KS trends
* Segment default rates
* Portfolio exposure
* Stress scenarios

---

#  What I Learned

This project helped me understand that credit-risk modelling is not simply about training a classifier.

The important part is connecting the model to the **risk-management decision process**.

I learned how to move from:

```text
Raw Lending Data
        ↓
Data Quality
        ↓
Risk Segmentation
        ↓
Feature Engineering
        ↓
Probability of Default
        ↓
Risk Ranking
        ↓
Risk Bands
        ↓
Stress Testing
        ↓
Business Decision
```

I also learned why an interpretable model can be valuable in a regulated environment where model outputs need to be explained rather than treated as a black box.

---

#  Conclusion

This project demonstrates an end-to-end **credit risk analytics lifecycle** for a retail lending portfolio.

The logistic regression model achieved a **ROC-AUC of 0.86** and **KS statistic of 0.59**, demonstrating strong discriminatory power while maintaining interpretability.

More importantly, the model's predictions were translated into five risk bands, with observed default rates increasing from **2.5% in the lowest-risk band to 68.9% in the highest-risk band**.

The addition of portfolio stress testing provides another layer of risk analysis by showing how predicted default risk could change under adverse interest-rate and affordability conditions.

Overall, the project demonstrates how credit-risk analytics can combine:

**Data Quality → Risk Segmentation → Predictive Modelling → Model Validation → Risk Bands → Stress Testing → Business Recommendations**

Before production use, additional validation such as out-of-time testing, calibration, fairness assessment, population-stability monitoring, and challenger-model comparison would be required.

---

#  Author

## Sakshi Chore

**Aspiring Data Analyst | Credit Risk & Data Science Enthusiast**

### Core Skills

`Python` • `SQL` • `Pandas` • `NumPy` • `Scikit-learn` • `Machine Learning` • `Credit Risk Analytics` • `Data Analysis` • `Data Visualization`

### GitHub

[GitHub Profile](https://github.com/sakshi200-code)

### Project Repository

[Credit Risk RBI Project](https://github.com/sakshi200-code/credit-risk-rbi-project)

---





