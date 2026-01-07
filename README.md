📊 AI Customer Retention & Revenue Intelligence
📝 Executive Summary

AI Customer Retention & Revenue Intelligence is a production-ready, end-to-end AI system designed to prevent customer churn and maximize revenue for subscription-based businesses. It integrates:

Advanced Exploratory Data Analysis (EDA)

Feature Engineering

Supervised & Unsupervised Machine Learning

Explainable AI (XAI)

Business-driven Recommendations

The system is fully Flask-deployable and comes with saved ML pipelines for real-world deployment.

Key Business Questions Addressed:

Who will churn? (Classification)

How valuable are they? (Revenue Regression)

What type of customer are they? (Clustering & Segmentation)

What action should the business take? (Recommendation Engine)

1️⃣ Project Objective

Analyze customer behavior, predict churn risk, estimate revenue impact, segment customers into actionable groups, and provide explainable, data-driven recommendations to reduce churn and increase profitability.

2️⃣ Dataset Overview

Source: Telco Customer Churn Dataset

Customers: ~7,000

Original Features: 21

Final ML-Ready Features: 38+

Target Variable: Churn → 1 (Churned), 0 (Retained)

Data Cleaning:

No duplicate records

TotalCharges converted to numeric

Missing values handled with safe imputation

✅ Dataset is clean, consistent, and production-ready.

3️⃣ Exploratory Data Analysis (EDA)

Performed to uncover churn patterns and revenue signals.

Key Insights:

Higher churn among month-to-month contracts, electronic check users, low-tenure customers, and high monthly charge customers

Visualization Techniques:

Count plots (Churn vs Gender, Payment Method)

Box plots (Tenure, Total Charges)

Histograms (Monthly & Total Charges)

Line plots (Tenure vs Charges)

Scatter plots (Spending relationships)

Pie charts (Churn distribution)

📌 Demonstrates strong analytical thinking and business storytelling.

4️⃣ Feature Engineering (Key Strength ⭐)
Feature	Business Meaning
tenure_group	Customer lifecycle stage (New / Mid / Loyal)
is_long_term_contract	Contract stability indicator
is_auto_payment	Payment reliability
num_services	Engagement level
avg_monthly_spend	Spending efficiency
is_single	Dependency & churn risk proxy

📌 Features reflect domain understanding, not just preprocessing.

5️⃣ Data Encoding & Scaling

Binary encoding for Yes/No features

One-Hot Encoding for categorical services, contracts, payment methods, tenure groups

Dummy variable trap avoided (drop_first=True)

StandardScaler applied where mathematically required

✅ ML-optimized dataset generated: customer_churn_ml_ready.csv

6️⃣ Churn Prediction – Classification Module

Model Comparison:

Model	Accuracy
Logistic Regression	~78%
Decision Tree	~76%
Random Forest	~79%
Gradient Boosting	~81%
XGBoost	Comparable

Selected Model: Gradient Boosting Classifier

Reasons:

Balanced precision & recall

Strong generalization

Clear feature importance

Advanced Techniques:

SMOTE for class imbalance

GridSearchCV (5-fold cross-validation)

Performance:

Test Accuracy: ~81%

CV Mean Accuracy: ~79%

Recall (Churn): ~71%

📌 Prioritizes identifying churners early for business action.

7️⃣ Revenue Prediction – Regression Module

Objective: Predict monthly revenue per customer to quantify financial risk.

Best Model: Linear Regression Pipeline

R² Score: 0.9989

MAE: ~0.75

📌 Enables revenue-aware churn prioritization.

8️⃣ Customer Segmentation – Clustering Module

Method: K-Means clustering

Optimal clusters via Elbow Method

Final choice: 4 clusters

Cluster Interpretation:

Cluster	Description
0	Low-tenure, low-spend, high risk
1	Mid-tenure, moderate value
2	Loyal, high-value customers
3	Price-sensitive but stable

Features Used: Behavioral, financial, and stability indicators (no target leakage)
PCA Visualization: Clear cluster separation for actionable segmentation.

9️⃣ Explainable AI (XAI)

SHAP – Churn Model: Explains why predictions are made; top churn drivers: Tenure, Monthly Charges, Contract Type, Avg Monthly Spend

Permutation Importance – Revenue Model: Identifies true revenue drivers; confirms multi-service, long-term customers generate higher value

📌 Ensures trust, transparency, and interpretability.

🔟 Recommendation Engine

Rule-based business logic converts predictions into actions:

Condition	Action
High churn + high charges	Offer Discount
High churn + low charges	Loyalty Reward
Low churn + high value	Upsell
Others	No Action

📌 Bridges ML outputs with real business strategy.

1️⃣1️⃣ Deployment & System Architecture

Saved Models:

churn_model.pkl

revenue_model.pkl

customer_segmentation_kmeans.pkl

pca_model.pkl

scaler_kmeans.pkl

Architecture:

Layer	Description
Data	Feature engineering & encoding
ML	Classification, Regression, Clustering
XAI	SHAP & Permutation Importance
Visualization	PCA, SHAP, Confusion Matrix
Intelligence	Recommendation Engine
Deployment	Flask-ready, CSV & manual input
🚀 Final Assessment

AI Customer Retention & Revenue Intelligence is a flagship-level AI system demonstrating:

Strong ML fundamentals

Business-driven thinking

Explainable AI

Production readiness

Real-world applicability

Suitable For:

AI / ML Engineer roles

Data Scientist positions

Portfolio & GitHub showcase

Technical interviews

🎯 Interview One-Line Summary:

“I built a production-ready AI customer intelligence system that predicts churn, estimates revenue impact, segments customers, explains model decisions, and converts predictions into actionable business strategies.”

💻 Tech Stack & Requirements

Python Libraries:

beautifulsoup4==4.14.2
blinker==1.9.0
certifi==2025.10.5
charset-normalizer==3.4.4
click==8.3.0
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
Flask==3.1.2
fonttools==4.60.1
idna==3.11
imbalanced-learn==0.14.0
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.5.2
kiwisolver==1.4.9
llvmlite==0.46.0b1
MarkupSafe==3.0.3
matplotlib==3.10.7
numba==0.63.0b1
numpy==2.3.4
packaging==25.0
pandas==2.3.3
pillow==12.0.0
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
requests==2.32.5
scikit-learn==1.7.2
scipy==1.16.2
seaborn==0.13.2
setuptools==80.9.0
shap==0.50.0
six==1.17.0
slicer==0.0.8
soupsieve==2.8
threadpoolctl==3.6.0
tqdm==4.67.1
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
Werkzeug==3.1.3
wheel==0.45.1
xgboost==3.1.1

📂 Flask App Overview

CSV Upload: Batch predictions + SHAP & PCA visuals

Manual Input: Single customer prediction + SHAP & PCA plots

Download Results: CSV download for batch predictions

Built with Flask, Pandas, Scikit-learn, SHAP, Seaborn, Matplotlib

Author: Musfira Mubeen
Role Target: AI / Machine Learning Engineer
