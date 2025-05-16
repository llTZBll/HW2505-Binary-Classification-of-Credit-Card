import os
from datetime import datetime

# 项目根目录 - 修正为向上三级目录（因为config.py在src/下）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'train.csv')  # 原始训练数据路径
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')  # 测试数据路径

# 结果输出路径
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')

# 确保所有目录存在
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 文件路径
MODEL_PATH = os.path.join(MODELS_DIR, 'final_model.txt')  # 模型保存路径
PREDICTION_OUTPUT_PATH = os.path.join(REPORTS_DIR, 'predictions.csv')  # 预测结果路径
OPTIMIZATION_RESULT_PATH = os.path.join(REPORTS_DIR, 'optimization_results.csv')  # 优化结果路径
LOG_PATH = os.path.join(LOGS_DIR, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')  # 日志路径

# 模型参数
LGB_PARAMS = {
    'kfold': 3,
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
    'max_evals': 50
}

# 固定参数
FIXED_PARAMS = {
    'learning_rate': 0.05,
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': 42,
    'is_unbalance': True,
    'force_col_wise': True
}
