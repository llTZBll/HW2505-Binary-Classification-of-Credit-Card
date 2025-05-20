import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os
import sys
import logging
from datetime import datetime

# 添加路径，确保可以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置和模块
from config import *
from utils.preprocessing import *
from utils.evaluation import *
from models.lgb_optimizer import LGBBayesianOptimizer

# 配置日志
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():

    logger.info("开始执行主程序")

    ######################   数据预处理   #######################################
    # 数据加载与预处理
    logger.info("开始数据加载与预处理")
    df = load_data(DATA_PATH)
    print(f"数据加载完成，形状: {df.shape}")

    # 删除ID列
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        print("已删除ID列")

    # 查看缺失值
    print("缺失值统计:")
    print(df.isnull().sum())

    df = fill_categorical_nan_with_mode(df)
    df = label_encode_categorical_features(df)

    print("\n数据预处理完成，前5行:")
    print(df.head())

    # 准备特征和标签
    selected_features, label = prepare_data(df, 'Is_Lead')

    # 创建交叉验证
    pdf_train = create_cv_folds(df, selected_features, label, n_splits=LGB_PARAMS['kfold'])
    print("交叉验证数据准备完成")

    # 划分训练集和测试集
    X = pdf_train[selected_features]
    y = pdf_train[label]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    ######################  超参数优化  ##############################################

    # 初始化优化器
    logger.info("开始模型参数优化")
    optimizer = LGBBayesianOptimizer(
        result_path=OPTIMIZATION_RESULT_PATH,
        kfold=LGB_PARAMS['kfold'],
        n_estimators=LGB_PARAMS['n_estimators'],
        early_stopping_rounds=LGB_PARAMS['early_stopping_rounds']
    )

    # 加载数据并优化
    optimizer.load_data(pdf_train, selected_features, label)
    optimizer.optimize(max_evals=LGB_PARAMS['max_evals'])

    # 读取优化结果
    results = pd.read_csv(OPTIMIZATION_RESULT_PATH)

    # 找到最佳参数组合
    best_result = results.loc[results['valid_auc'].idxmax()]
    print("最佳验证AUC:", best_result['valid_auc'])
    print("\n最佳参数:")
    print(best_result['params'])

    # 绘制AUC变化趋势
    plot_optimization_progress(OPTIMIZATION_RESULT_PATH)
    logger.info("优化过程图表已保存")

#########################  训练最终模型  ############################################

    # 从最佳结果中提取参数
    best_params = eval(best_result['params'])
    best_params.update(FIXED_PARAMS)

    # 准备完整训练集和测试集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 使用最佳参数训练最终模型
    logger.info("开始训练最终模型")
    final_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        num_boost_round=int(best_result['best_round'] * 1.1),  # 比平均最佳轮数多10%
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(50)
        ]
    )

    # 保存模型
    remark = "LGBM_max_evals="+ str(LGB_PARAMS['max_evals'])
    get_model_path(remark)

    MODEL_PATH = get_model_path(remark)  # 模型保存路径
    final_model.save_model(MODEL_PATH)
    logger.info(f"模型已保存至 {MODEL_PATH}")

    # 评估模型
    test_auc = evaluate_model(final_model, X_test, y_test)
    logger.info(f"最终模型测试AUC: {test_auc:.4f}")

    # 特征重要性
    feature_importance = plot_feature_importance(final_model, selected_features)
    logger.info("特征重要性分析已完成")

    print("\n所有任务已完成!")
    logger.info("主程序执行完成")


if __name__ == "__main__":
    main()