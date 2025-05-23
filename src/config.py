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
def get_model_path(remark=''):
    """生成带有时间和备注的模型保存路径"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if remark:
        filename = f"model_{current_time}_{remark}.txt"
    else:
        filename = f"model_{current_time}.txt"
    return os.path.join(MODELS_DIR, filename)

#MODEL_PATH = get_model_path('final') # 模型保存路径
PREDICTION_OUTPUT_PATH = os.path.join(REPORTS_DIR, 'predictions_2.csv')  # 预测结果路径
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