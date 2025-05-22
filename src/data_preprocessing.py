import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from src.utils.preprocessing import label_encode_categorical_features

# 读取数据
df = pd.read_csv('../data/train.csv')


# 准备评估函数
def evaluate_imputation_method(df, impute_func):
    # 复制数据
    df_imputed = df.copy()

    # 应用缺失值填充方法
    df_imputed = impute_func(df_imputed)

    # 标签编码
    df_imputed = label_encode_categorical_features(df_imputed)

    # 划分训练集和测试集
    X = df_imputed.drop(['ID', 'Is_Lead'], axis=1)
    y = df_imputed['Is_Lead']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型并评估
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)


# 方法1：删除缺失值
def impute_by_deletion(df):
    return df.dropna(subset=['Credit_Product'])


# 方法2：填充为特定值
def impute_by_specific_value(df):
    df['Credit_Product'] = df['Credit_Product'].fillna('Unknown')
    return df


# 方法3：填充为众数
def impute_by_mode(df):
    mode = df['Credit_Product'].mode()[0]
    df['Credit_Product'] = df['Credit_Product'].fillna(mode)
    return df


# 方法4：基于其他特征的KNN填充
def impute_by_knn(df):
    # 对其他分类特征进行编码
    cat_cols = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code', 'Is_Active']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # 创建特征矩阵
    X = df.drop(['ID', 'Is_Lead', 'Credit_Product'], axis=1)
    y = df['Credit_Product']

    # 使用KNN填充
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)

    # 将填充好的数据转回DataFrame
    X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
    df['Credit_Product'] = y.fillna(X_imputed_df['Credit_Product'])

    return df


# 方法5：基于树模型的预测填充
def impute_by_model(df):
    # 分割有值和缺失值的样本
    df_with_value = df[df['Credit_Product'].notna()]
    df_missing = df[df['Credit_Product'].isna()]

    # 对其他分类特征进行编码
    cat_cols = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code', 'Is_Active']
    for col in cat_cols:
        le = LabelEncoder()
        df_with_value[col] = le.fit_transform(df_with_value[col])
        df_missing[col] = le.transform(df_missing[col])

    # 训练模型预测缺失值
    X_train = df_with_value.drop(['ID', 'Is_Lead', 'Credit_Product'], axis=1)
    y_train = df_with_value['Credit_Product']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 预测缺失值
    X_test = df_missing.drop(['ID', 'Is_Lead', 'Credit_Product'], axis=1)
    y_pred = model.predict(X_test)

    # 填充缺失值
    df_missing['Credit_Product'] = y_pred
    df_imputed = pd.concat([df_with_value, df_missing])

    return df_imputed


# 评估各种方法
methods = {
    '删除缺失值': impute_by_deletion,
    '填充特定值': impute_by_specific_value,
    '填充众数': impute_by_mode,
    'KNN填充': impute_by_knn,
    '模型预测填充': impute_by_model
}

results = {}
for name, method in methods.items():
    try:
        auc = evaluate_imputation_method(df, method)
        results[name] = auc
        print(f"{name}方法的AUC: {auc:.4f}")
    except Exception as e:
        print(f"{name}方法执行出错: {e}")
        results[name] = None

# 找出最佳方法
best_method = max(results, key=lambda k: results[k] if results[k] is not None else float('-inf'))
print(f"\n最佳填充方法: {best_method} (AUC: {results[best_method]:.4f})")