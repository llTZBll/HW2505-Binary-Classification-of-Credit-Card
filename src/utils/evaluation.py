import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb


def plot_optimization_progress(results_path):
    """绘制优化过程中的AUC变化"""
    results = pd.read_csv(results_path)

    plt.figure(figsize=(12, 6))
    plt.plot(results['iteration'], results['valid_auc'], label='Validation AUC')
    plt.plot(results['iteration'], results['train_auc'], label='Train AUC')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.title('Bayesian Optimization Progress')
    plt.legend()
    plt.grid()
    plt.savefig('../results/plots/optimization_progress.png')
    plt.close()


def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    test_pred = model.predict(X_test)
    test_auc = roc_auc_score(y_test, test_pred)
    print(f"最终模型测试AUC: {test_auc:.4f}")

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, test_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('../results/plots/roc_curve.png')
    plt.close()

    return test_auc


def plot_feature_importance(model, feature_names):
    """绘制特征重要性"""
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('../results/plots/feature_importance.png')
    plt.close()

    # 保存特征重要性数据
    feature_importance.to_csv('../results/reports/feature_importance.csv', index=False)

    return feature_importance