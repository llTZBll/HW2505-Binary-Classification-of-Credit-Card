import csv
import time
from timeit import default_timer as timer
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import lightgbm as lgb
from hyperopt.pyll.base import scope
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

class LGBBayesianOptimizer:
    def __init__(self, result_path, **kwargs):
        self.result_path = result_path
        self.iter = 0
        self.train_set = None
        self.kfold = kwargs.get('kfold', 3)
        self.n_estimators = kwargs.get('n_estimators', 1000)
        self.early_stopping_rounds = kwargs.get('early_stopping_rounds', 100)

        # 初始化结果文件
        with open(self.result_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'loss', 'train_auc', 'valid_auc',
                'params', 'iteration', 'train_time',
                'best_round', 'feature_importance'
            ])

    def load_data(self, df_data, feature_list, label):
        self.df_data = df_data.reset_index(drop=True)
        self.feature_list = feature_list
        self.label = label

    def objective(self, params):
        self.iter += 1
        start_time = timer()

        # 确保参数类型正确
        params = {
            'max_depth': int(params['max_depth']),
            'num_leaves': int(params['num_leaves']),
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
            'min_child_samples': int(params['min_child_samples']),
            'min_split_gain': params['min_split_gain'],
            'learning_rate': 0.05,
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True,
            'force_col_wise': True
        }

        # 存储每折的结果
        train_auc_scores = []
        valid_auc_scores = []
        best_rounds = []
        feature_importances = []

        for k in range(self.kfold):
            # 准备数据
            train_data = self.df_data[self.df_data[f"cv{k}"] == 'train']
            valid_data = self.df_data[self.df_data[f"cv{k}"] == 'valid']

            # 创建数据集
            lgb_train = lgb.Dataset(
                train_data[self.feature_list],
                train_data[self.label]
            )
            lgb_valid = lgb.Dataset(
                valid_data[self.feature_list],
                valid_data[self.label],
                reference=lgb_train
            )

            # 训练模型
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=['train', 'valid'],
                num_boost_round=self.n_estimators,
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0)  # 不打印日志
                ]
            )

            # 记录最佳轮数
            best_rounds.append(model.best_iteration)

            # 记录特征重要性
            feature_importances.append(model.feature_importance(importance_type='gain'))

            # 计算AUC
            train_pred = model.predict(train_data[self.feature_list])
            valid_pred = model.predict(valid_data[self.feature_list])

            train_auc = roc_auc_score(train_data[self.label], train_pred)
            valid_auc = roc_auc_score(valid_data[self.label], valid_pred)

            train_auc_scores.append(train_auc)
            valid_auc_scores.append(valid_auc)

        # 计算平均AUC
        avg_train_auc = np.mean(train_auc_scores)
        avg_valid_auc = np.mean(valid_auc_scores)

        # 我们的目标是最大化验证集AUC，所以loss是负的AUC
        loss = -avg_valid_auc

        # 计算平均最佳轮数
        avg_best_round = int(np.mean(best_rounds))

        # 计算平均特征重要性
        avg_feature_importance = np.mean(feature_importances, axis=0)

        # 记录结果
        with open(self.result_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                loss,
                avg_train_auc,
                avg_valid_auc,
                str(params),
                self.iter,
                timer() - start_time,
                avg_best_round,
                str(avg_feature_importance.tolist())
            ])

        # 打印进度
        if self.iter % 10 == 0:
            print(f"Iteration {self.iter}: Train AUC = {avg_train_auc:.4f}, Valid AUC = {avg_valid_auc:.4f}")

        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': params,
            'train_auc': avg_train_auc,
            'valid_auc': avg_valid_auc,
            'time': timer() - start_time
        }

    def optimize(self, max_evals=100):
        self.iter = 0

        # 定义参数空间 - 使用scope确保整数参数
        space = {
            'max_depth': scope.int(hp.quniform('max_depth', 3, 8, 1)),
            'num_leaves': scope.int(hp.quniform('num_leaves', 15, 150, 5)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(10)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(10)),
            'min_child_samples': scope.int(hp.quniform('min_child_samples', 10, 200, 5)),
            'min_split_gain': hp.uniform('min_split_gain', 0.0, 0.2)
        }

        # 运行优化
        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=Trials(),
            rstate=np.random.default_rng(42)
        )

        print("\n优化完成!")
        print("最佳参数索引:", best)

        return best