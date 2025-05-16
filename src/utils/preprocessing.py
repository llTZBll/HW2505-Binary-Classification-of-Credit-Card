import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

def load_data(file_path):
    """加载数据（自动处理相对路径）"""
    import os
    # 若路径不是绝对路径，则基于项目根目录补全
    if not os.path.isabs(file_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(project_root, file_path)
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """处理缺失值"""
    df['Credit_Product'].fillna(df['Credit_Product'].mode()[0], inplace=True)
    return df

def encode_categorical_features(df):
    """对类别特征进行编码"""
    cat_cols = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code', 'Credit_Product', 'Is_Active']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def create_cv_folds(df, selected_features, label, n_splits=3, random_state=42):
    """创建交叉验证折叠"""
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    pdf_train = df.reset_index(drop=True)
    for k, (itrain, ivalid) in enumerate(kf.split(pdf_train[selected_features], pdf_train[label])):
        pdf_train[f"cv{k}"] = None
        pdf_train.loc[itrain, f'cv{k}'] = 'train'
        pdf_train.loc[ivalid, f'cv{k}'] = 'valid'
    return pdf_train

def prepare_data(df, label):
    """准备特征和标签"""
    selected_features = [col for col in df.columns if col != label]
    return selected_features, label