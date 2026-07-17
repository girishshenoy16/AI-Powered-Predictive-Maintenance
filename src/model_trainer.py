from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def train_and_evaluate(train_df, test_df):
    if train_df is None or test_df is None:
        print("Missing training or testing data. Aborting training.")
        return
        
    features = [col for col in train_df.columns if col not in ['engine_id', 'cycle', 'RUL', 'label']]
    
    X_train = train_df[features]
    y_class_train = train_df['label']
    y_reg_train = train_df['RUL']
    
    X_test = test_df[features]
    y_class_test = test_df['label']
    y_reg_test = test_df['RUL']
    
    print("\n--- Training Models ---")
    print("Training XGBoost Classifier...")
    xgb_class = XGBClassifier(eval_metric='logloss')
    xgb_class.fit(X_train, y_class_train)
    
    print("Training XGBoost Regressor...")
    xgb_reg = XGBRegressor(n_estimators=100, random_state=42)
    xgb_reg.fit(X_train, y_reg_train)
    
    # Evaluate
    class_preds = xgb_class.predict(X_test)
    reg_preds = xgb_reg.predict(X_test)
    
    print("\n--- Evaluation on True Test Set ---")
    print(f"Classifier Accuracy: {accuracy_score(y_class_test, class_preds):.4f}")
    print(f"Regressor Mean Absolute Error (MAE): {mean_absolute_error(y_reg_test, reg_preds):.2f} cycles")
    
    # Generate and Save Plots
    print("\nSaving evaluation plots to 'outputs/' directory...")
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_class_test, class_preds), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Failure Imminent'], yticklabels=['Normal', 'Failure Imminent'])
    plt.title('Classifier Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=300)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_class_test, xgb_class.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classifier ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=300)
    plt.close()
    
    # 3. True vs Predicted RUL
    plt.figure(figsize=(6, 5))
    plt.scatter(y_reg_test, reg_preds, alpha=0.3, color='purple')
    plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    plt.xlabel('True RUL (Cycles)')
    plt.ylabel('Predicted RUL (Cycles)')
    plt.title('Regressor: True vs Predicted RUL')
    plt.tight_layout()
    plt.savefig("outputs/rul_predictions.png", dpi=300)
    plt.close()
    
    print("Plots saved successfully.")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_class, "models/xgb_classifier.pkl")
    joblib.dump(xgb_reg, "models/xgb_regressor.pkl")
    print("Models trained and saved to models/ directory.")
