import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import joblib
import os
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import ssl
import urllib.request
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for 400 dpi
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400

def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        phishing = fetch_ucirepo(id=327)
        X = phishing.data.features
        y = phishing.data.targets['result']
    except Exception as e:
        print("ucimlrepo failed, downloading directly...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        response = urllib.request.urlopen(url, context=ctx)
        data, meta = arff.loadarff(response)
        df = pd.DataFrame(data)
        
        # arff loaded as bytes, decode to string and convert to int
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8').astype(int)
        
        X = df.drop(columns=['Result'])
        y = df['Result']
        
    print(f"Target value counts before mapping:\n{y.value_counts()}")
    # Map target: in UCI dataset, 1 is legitimate and -1 is phishing.
    # We will map -1 to 1 (phishing) and 1 to 0 (legitimate) to make phishing the positive class
    y = (y == -1).astype(int)
    print(f"Target value counts after mapping (1=Phishing):\n{y.value_counts()}")
    
    return X, y

def main():
    print("Loading data...")
    X, y = load_data()
    
    # 1. Stratified Split 80/20
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # 2. SMOTE on Training Only
    print("Applying SMOTE...")
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"Training set after SMOTE:\n{y_train_bal.value_counts()}")
    
    # 3. Feature Selection (Permutation Importance on baseline model)
    print("Selecting top 14 features using Permutation Importance...")
    base_clf = XGBClassifier(random_state=42, eval_metric='logloss')
    base_clf.fit(X_train_bal, y_train_bal)
    
    perm_importance = permutation_importance(base_clf, X_train_bal, y_train_bal, n_repeats=5, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    top_14_idx = sorted_idx[:14]
    
    top_14_features = X.columns[top_14_idx].tolist()
    print(f"Top 14 features: {top_14_features}")
    
    X_train_sel = X_train_bal.iloc[:, top_14_idx]
    X_test_sel = X_test.iloc[:, top_14_idx]
    
    # 4. XGBoost with GridSearchCV
    print("Training XGBoost with GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.1, 0.2]
    }
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    grid = GridSearchCV(xgb, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
    grid.fit(X_train_sel, y_train_bal)
    
    best_clf = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    
    # Save Models
    print("Serializing and saving model artifacts...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_clf, 'models/xgboost_model.joblib')
    # Fit a scaler specifically for the 14 features for inference
    scaler_14 = StandardScaler()
    scaler_14.fit(X_train_sel)
    joblib.dump(scaler_14, 'models/scaler.joblib')
    joblib.dump(top_14_features, 'models/top_14_features.joblib')
    
    # 5. Evaluate Full Proposed Model
    print("Evaluating Proposed Model...")
    y_pred = best_clf.predict(X_test_sel)
    y_prob = best_clf.predict_proba(X_test_sel)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob),
        'APS': average_precision_score(y_test, y_prob)
    }
    
    print("\n--- Full Proposed Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # --- ABLATION STUDY ---
    print("\nRunning Ablation Study...")
    # Model 1: Baseline (No SMOTE, No FeatSel, Default XGBoost)
    base_xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    base_xgb.fit(X_train, y_train)
    y_pred_b1 = base_xgb.predict(X_test)
    y_prob_b1 = base_xgb.predict_proba(X_test)[:, 1]
    b1_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_b1),
        'Recall': recall_score(y_test, y_pred_b1),
        'AUC': roc_auc_score(y_test, y_prob_b1)
    }
    
    # Model 2: XGBoost + SMOTE (No FeatSel)
    base_xgb_smote = XGBClassifier(random_state=42, eval_metric='logloss')
    base_xgb_smote.fit(X_train_bal, y_train_bal)
    y_pred_b2 = base_xgb_smote.predict(X_test)
    y_prob_b2 = base_xgb_smote.predict_proba(X_test)[:, 1]
    b2_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_b2),
        'Recall': recall_score(y_test, y_pred_b2),
        'AUC': roc_auc_score(y_test, y_prob_b2)
    }
    
    print("\n--- Ablation Metrics ---")
    print(f"Baseline: Acc={b1_metrics['Accuracy']:.4f}, Rec={b1_metrics['Recall']:.4f}, AUC={b1_metrics['AUC']:.4f}")
    print(f"+SMOTE: Acc={b2_metrics['Accuracy']:.4f}, Rec={b2_metrics['Recall']:.4f}, AUC={b2_metrics['AUC']:.4f}")
    
    # Visualizations
    print("Generating visualizations...")
    
    # 1. ROC Curve
    fpr1, tpr1, _ = roc_curve(y_test, y_prob_b1)
    fpr2, tpr2, _ = roc_curve(y_test, y_prob_b2)
    fpr3, tpr3, _ = roc_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, label=f'Baseline (AUC={b1_metrics["AUC"]:.3f})', linestyle=':')
    plt.plot(fpr2, tpr2, label=f'XGBoost + SMOTE (AUC={b2_metrics["AUC"]:.3f})', linestyle='--')
    plt.plot(fpr3, tpr3, label=f'Proposed (AUC={metrics["AUC"]:.3f})', linestyle='-', color='#2563eb')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=400)
    plt.close()
    
    # 2. Ablation Performance Bar Chart
    labels = ['Baseline', '+SMOTE', 'Proposed']
    accs = [b1_metrics['Accuracy'], b2_metrics['Accuracy'], metrics['Accuracy']]
    recs = [b1_metrics['Recall'], b2_metrics['Recall'], metrics['Recall']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#1b3a6b')
    rects2 = ax.bar(x + width/2, recs, width, label='Recall', color='#2563eb')
    
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='upper left')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('ablation_bar.png', dpi=400)
    plt.close()
    
    # 3. SHAP Summary Plot
    explainer = shap.TreeExplainer(best_clf)
    shap_values = explainer(X_test_sel)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_sel, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    # 4. SHAP Waterfall for a single phishing instance
    phishing_indices = np.where(y_test == 1)[0]
    sample_idx = phishing_indices[0] # taking the first phishing sample
    
    plt.figure()
    # Provide the feature names and base values correctly
    # shap_values from TreeExplainer might be returned as an Explanation object in newer shap versions
    if isinstance(shap_values, shap.Explanation):
        shap.plots.waterfall(shap_values[sample_idx], show=False)
    else:
        # Fallback for older SHAP versions
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[sample_idx], feature_names=X_test_sel.columns, show=False)
    plt.tight_layout()
    plt.savefig('shap_waterfall.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    print("Done! Visualizations saved in the current directory.")

if __name__ == '__main__':
    main()
