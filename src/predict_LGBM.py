import pandas as pd
import lightgbm as lgb
from config import TEST_DATA_PATH, PREDICTION_OUTPUT_PATH
from utils.preprocessing import (
    load_data,
    fill_categorical_nan_with_mode,
    label_encode_categorical_features,
    prepare_data, fill_categorical_nan_with_unknown
)
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """加载训练好的LightGBM模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练模型")
    logger.info(f"从 {model_path} 加载模型")
    return lgb.Booster(model_file=model_path)


def make_predictions():
    """执行完整的预测流程"""

    # 1. 加载模型
    MODEL_PATH ="../results/models/model_20250522_165129_LGBM_max_evals=50.txt"
    model = load_model(MODEL_PATH)

    # 2. 加载测试数据
    logger.info(f"从 {TEST_DATA_PATH} 加载测试数据")
    df_test = load_data(TEST_DATA_PATH)

    # 3. 保存原始ID列位置
    original_id_column = None
    if 'ID' in df_test.columns:
        original_id_column = 'ID'
    elif df_test.columns[0] == 'ID':  # 处理无列名但第一列是ID的情况
        original_id_column = df_test.columns[0]
    else:
        logger.warning("未找到ID列，将生成默认ID")

    # 4. 应用与训练相同的预处理
    logger.info("开始数据预处理")
    df_test = fill_categorical_nan_with_unknown(df_test)
    df_test = label_encode_categorical_features(df_test)

    # 5. 准备特征 (使用与训练相同的特征列)
    selected_features, _ = prepare_data(df_test, 'Is_Lead')  # 虽然测试集没有Is_Lead，但保持接口一致

    # 6. 确保ID列不包含在特征中
    if original_id_column and original_id_column in selected_features:
        selected_features.remove(original_id_column)
        logger.info(f"从特征中移除ID列: {original_id_column}")

    # 7. 获取最终的特征矩阵
    X_test = df_test[selected_features]

    # 8. 重新获取ID列（预处理后可能位置改变）
    if original_id_column:
        ids = df_test[original_id_column]
    else:
        ids = pd.RangeIndex(start=1, stop=len(df_test) + 1)

    # 9. 进行预测
    logger.info("开始预测")
    predictions = model.predict(X_test)

    # 将概率转换为二分类标签（0/1）
    threshold = 0.5  # 阈值，可根据业务需求调整
    binary_predictions = (predictions >= threshold).astype(int)

    # 10. 保存结果 (格式与train.csv相同，包含ID和预测的Is_Lead列)
    output_df = pd.DataFrame({
        'ID': ids,
        'probability':predictions,
        'Is_Lead': binary_predictions
    })

    # 创建输出目录如果不存在
    os.makedirs(os.path.dirname(PREDICTION_OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(PREDICTION_OUTPUT_PATH, index=False)
    logger.info(f"预测结果已保存至 {PREDICTION_OUTPUT_PATH}")

    return output_df


if __name__ == "__main__":
    make_predictions()