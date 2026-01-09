# 📊 AI Customer Retention & Revenue Intelligence
### *End-to-End AI System for Churn Prediction & Revenue Optimization*

---

## 📌 Project Overview
**AI Customer Retention & Revenue Intelligence** is a production-ready system designed to prevent customer churn and maximize revenue for subscription-based businesses. This project goes beyond simple prediction by integrating **Classification, Regression, Clustering, and Explainable AI (XAI)** into a single unified pipeline.

Instead of just identifying "who" will leave, this system explains **why**, quantifies the **financial impact**, and provides **automated business recommendations**.

---

## 🚀 Key Business Intelligence
* **Churn Prediction:** Identifying high-risk customers before they leave.
* **Revenue Forecasting:** Estimating the monthly revenue value per customer.
* **Customer Segmentation:** Grouping customers by behavior and value.
* **Actionable Insights:** Converting ML outputs into "Next-Best-Action" strategies.
* **Explainability:** Using **SHAP** to make black-box models transparent.

---

## 🧠 System Architecture

### 1️⃣ Churn Classification (Gradient Boosting)
* **Goal:** Predict probability of churn (1 or 0).
* **Performance:** ~81% Accuracy | ~71% Recall.
* **Handling Imbalance:** Optimized using **SMOTE** and **GridSearchCV**.

### 2️⃣ Revenue Regression (Linear Pipeline)
* **Goal:** Quantify the financial value of each customer.
* **Performance:** R² Score: **0.9989** | MAE: ~0.75.

### 3️⃣ Customer Clustering (K-Means)
* **Goal:** Group customers into 4 actionable segments.
* **Technique:** Optimized via Elbow Method and visualized using **PCA**.

### 4️⃣ Explainable AI (SHAP)
* **Goal:** Global and local interpretability to understand top churn drivers (Tenure, Contract Type, Monthly Charges).

---

## 🛠️ Feature Engineering (Domain Driven)
| Feature | Business Meaning |
| :--- | :--- |
| **tenure_group** | Customer lifecycle stage (New / Mid / Loyal) |
| **is_long_term** | Contract stability indicator |
| **num_services** | Engagement level (Total services used) |
| **avg_monthly_spend** | Spending efficiency & value proxy |

---

## 📊 Business Strategy Matrix
The system automatically generates recommendations based on model outputs:

| Risk Level | Customer Value | Recommended Action |
| :--- | :--- | :--- |
| 🔴 **High** | 💰 **High** | Personal VIP Discount / Retention Call |
| 🔴 **High** | 📉 **Low** | Loyalty Rewards / Feedback Survey |
| 🟢 **Low** | 💰 **High** | Upsell Premium Services |
| 🟢 **Low** | 📉 **Low** | Standard Engagement |

---

## 💻 Tech Stack
* **Core:** Python, Pandas, Scikit-Learn
* **ML/NLP:** XGBoost, Imbalanced-Learn, SHAP
* **Web:** Flask (Deployment Ready)
* **Visualization:** Seaborn, Matplotlib, PCA

---

## 📂 Project Structure
* `app.py`: Flask web application.
* `models/`: Saved `.pkl` files (Churn, Revenue, Scalers, PCA).
* `notebooks/`: Advanced EDA and Model Training scripts.
* `data/`: Cleaned, ML-ready datasets.

---

## ⚠️ Business Impact
> "I built this system to bridge the gap between raw data and executive decision-making. By combining churn risk with revenue impact, businesses can prioritize retention efforts where they matter most financially."

---

## 👩‍💻 Author
**Musfira Mubeen** *Aspiring AI/ML Engineer & Data Scientist*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/musfira-mubeen/)
[![Portfolio](https://img.shields.io/badge/Portfolio-View-green?style=flat&logo=github)](https://github.com/mubeen20012)

⭐ *If you find this project useful for your business or portfolio, feel free to star the repository!*
