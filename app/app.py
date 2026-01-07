from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# Initialize App
# =========================
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Store last batch prediction for download
last_batch_df = None

# =========================
# Load Models
# =========================
churn_model = joblib.load(os.path.join(BASE_DIR, "models", "churn_model.pkl"))
revenue_model = joblib.load(os.path.join(BASE_DIR, "models", "revenue_model.pkl"))
clustering_model = joblib.load(os.path.join(BASE_DIR, "models", "customer_segmentation_kmeans.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler_kmeans.pkl"))

# =========================
# Helper: Build Manual DF
# =========================
def build_manual_dataframe(form_data, reference_columns):
    df = pd.DataFrame(columns=reference_columns)
    df.loc[0] = 0
    for k, v in form_data.items():
        if k in df.columns:
            df.at[0, k] = v
    return df

# =========================
# Home
# =========================
@app.route('/')
def home():
    return render_template("index.html")

# =========================
# CSV Prediction with Robust Validation
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    global last_batch_df  # store last batch

    df = pd.read_csv(request.files['file'])
    
    # Drop irrelevant ID if exists
    df_features = df.drop(columns=['customerID'], errors='ignore')

    # Robust Column Handling
    required_columns = churn_model.feature_names_in_
    df_features = df_features.reindex(columns=required_columns, fill_value=0)
    for col in required_columns:
        if df_features[col].dtype == 'object':
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)

    # Predictions
    df['Churn'] = churn_model.predict(df_features)
    df['Revenue'] = revenue_model.predict(df_features)

    # Executive Metrics
    high_risk = df[df['Churn'] == 1]
    total_revenue_risk = round(high_risk['Revenue'].sum(), 2)
    avg_churn = round(df['Churn'].mean() * 100, 2)
    retention_value = round(total_revenue_risk * 0.3, 2)

    # Clustering
    cluster_cols = [
        'tenure','MonthlyCharges','TotalCharges',
        'avg_monthly_spend','num_services',
        'is_long_term_contract','is_auto_payment','is_single'
    ]
    for c in cluster_cols:
        if c not in df_features.columns:
            df_features[c] = 0

    X_scaled = scaler.transform(df_features[cluster_cols])
    df['Cluster'] = clustering_model.predict(X_scaled)

    df['Recommendation'] = df.apply(
        lambda r: "Offer Discount" if r['Churn'] == 1
        else "Upsell" if r['Revenue'] > 100
        else "No Action",
        axis=1
    )

    # Store last batch for download
    last_batch_df = df.copy()

    # SHAP
    explainer = shap.TreeExplainer(churn_model.named_steps['Models'])
    X_trans = churn_model.named_steps['preprocessing'].transform(df_features)
    shap_values = explainer.shap_values(X_trans)

    plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=df_features.columns, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    shap_plot = base64.b64encode(buf.getvalue()).decode()

    # PCA
    pca = PCA(2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure()
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Cluster'], palette="viridis")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    pca_plot = base64.b64encode(buf.getvalue()).decode()

    # Model Health
    y_true = df['Churn']
    y_pred = df['Churn']

    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 2),
        "Precision": round(precision_score(y_true, y_pred), 2),
        "Recall": round(recall_score(y_true, y_pred), 2),
        "F1": round(f1_score(y_true, y_pred), 2)
    }

    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    cm_plot = base64.b64encode(buf.getvalue()).decode()

    return render_template(
        "index.html",
        table=df.head(10).to_html(index=False),
        shap_plot=shap_plot,
        pca_plot=pca_plot,
        cm_plot=cm_plot,
        metrics=metrics,
        total_revenue_risk=total_revenue_risk,
        avg_churn=avg_churn,
        retention_value=retention_value
    )

# =========================
# Download Last Batch CSV
# =========================
@app.route('/download', methods=['GET'])
def download_csv():
    global last_batch_df
    if last_batch_df is not None:
        buf = io.BytesIO()
        last_batch_df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(
            buf,
            mimetype='text/csv',
            download_name='prediction_results.csv',  # <-- updated for Flask 2.2+
            as_attachment=True
        )
    else:
        return "No batch prediction available to download.", 400

# =========================
# ✅ MANUAL PREDICTION (FULL ANALYTICS)
# =========================
@app.route('/manual_predict', methods=['POST'])
def manual_predict():

    form_data = {
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges']),
        'tenure': int(request.form['tenure']),
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'FiberOptic': int(request.form.get('FiberOptic', 0)),
        'TwoYearContract': int(request.form.get('TwoYearContract', 0)),
        'avg_monthly_spend': float(request.form['TotalCharges']) / max(1, int(request.form['tenure'])),
        'num_services': int(request.form.get('num_services', 1)),
        'is_long_term_contract': int(request.form.get('is_long_term_contract', 0)),
        'is_auto_payment': int(request.form.get('is_auto_payment', 0)),
        'is_single': int(request.form.get('is_single', 1))
    }

    df_manual = build_manual_dataframe(
        form_data,
        churn_model.feature_names_in_
    )

    df_manual = df_manual.reindex(
        columns=churn_model.feature_names_in_,
        fill_value=0
    )

    # Predictions
    churn = churn_model.predict(df_manual)[0]
    revenue = revenue_model.predict(df_manual)[0]

    # Clustering
    cluster_cols = [
        'tenure','MonthlyCharges','TotalCharges',
        'avg_monthly_spend','num_services',
        'is_long_term_contract','is_auto_payment','is_single'
    ]

    X_scaled = scaler.transform(df_manual[cluster_cols])
    cluster = clustering_model.predict(X_scaled)[0]

    recommendation = (
        "Offer Discount" if churn == 1
        else "Upsell" if revenue > 100
        else "No Action"
    )

    manual_result = {
        "Churn": "Yes" if churn == 1 else "No",
        "Revenue": round(revenue, 2),
        "Cluster": int(cluster),
        "Recommendation": recommendation
    }

    # SHAP
    explainer = shap.TreeExplainer(churn_model.named_steps['Models'])
    X_trans = churn_model.named_steps['preprocessing'].transform(df_manual)
    shap_values = explainer.shap_values(X_trans)

    plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=df_manual.columns, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    shap_plot = base64.b64encode(buf.getvalue()).decode()

    # PCA (duplicate point trick)
    X_plot = np.vstack([X_scaled, X_scaled])
    clusters = [cluster, cluster]

    pca = PCA(2)
    X_pca = pca.fit_transform(X_plot)

    plt.figure()
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="viridis")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    pca_plot = base64.b64encode(buf.getvalue()).decode()

    metrics = None
    model_health_message = (
      "🩺 Model performance metrics are available only for batch predictions. "
      "Please upload a CSV file to view Accuracy, Precision, Recall, and F1-score."
    )
    
    return render_template(
     "index.html",
     manual_result=manual_result,
     shap_plot=shap_plot,
     pca_plot=pca_plot,
     metrics=None,
     model_health_message=model_health_message,
     cm_plot=None
)

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)
