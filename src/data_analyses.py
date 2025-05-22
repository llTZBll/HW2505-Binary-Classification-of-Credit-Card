import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency

# 读取数据集
df = pd.read_csv('../data/train.csv')

# 1. 统计缺失值比例
missing_ratio = df['Credit_Product'].isnull().mean()
print(f"Credit_Product缺失比例: {missing_ratio:.2%}")

# 2. 分类变量与缺失值的关系（卡方检验）
categorical_cols = ['Gender', 'Occupation', 'Channel_Code', 'Is_Active']
for col in categorical_cols:
    # 创建列联表
    contingency_table = pd.crosstab(df[col], df['Credit_Product'].isnull())
    # 卡方检验
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    # 输出结果（保留两位小数）
    print(f"\n{col}与Credit_Product缺失的卡方检验:")
    print(f"卡方值: {chi2:.2f}, p值: {p:.4f}")
    if p < 0.05:
        print(f"* {col}与Credit_Product缺失存在显著关联（拒绝原假设）")
    else:
        print(f"* {col}与Credit_Product缺失无显著关联（接受原假设）")

# 3. 连续变量与缺失值的关系（t检验）
continuous_cols = ['Age', 'Vintage', 'Avg_Account_Balance']
for col in continuous_cols:
    # 分组：缺失组与非缺失组
    missing_group = df[df['Credit_Product'].isnull()][col]
    non_missing_group = df[~df['Credit_Product'].isnull()][col]

    # 检查数据是否符合正态分布
    _, p_norm_missing = stats.normaltest(missing_group.dropna())
    _, p_norm_non_missing = stats.normaltest(non_missing_group.dropna())

    # 根据正态性选择检验方法
    if p_norm_missing < 0.05 or p_norm_non_missing < 0.05:
        # 非参数检验（Mann-Whitney U检验）
        _, p = stats.mannwhitneyu(missing_group.dropna(), non_missing_group.dropna())
        test_method = "Mann-Whitney U检验"
    else:
        # 参数检验（t检验）
        _, p = stats.ttest_ind(missing_group.dropna(), non_missing_group.dropna())
        test_method = "t检验"

    # 输出结果（保留两位小数）
    print(f"\n{col}与Credit_Product缺失的{test_method}:")
    print(f"p值: {p:.4f}")
    if p < 0.05:
        print(f"* {col}在缺失组和非缺失组之间存在显著差异（拒绝原假设）")
    else:
        print(f"* {col}在缺失组和非缺失组之间无显著差异（接受原假设）")

# 4. 基于缺失模式的相关性分析
df['Credit_Product_Missing'] = df['Credit_Product'].isnull().astype(int)
correlation = df.corr()['Credit_Product_Missing'].drop('Credit_Product_Missing')
print("\n缺失模式与其他数值变量的相关性:")
print(correlation.round(4))

# 5. 综合判断（简化版）
significant_associations = []
for col in categorical_cols:
    _, p, _, _ = chi2_contingency(pd.crosstab(df[col], df['Credit_Product_Missing']))
    if p < 0.05:
        significant_associations.append(col)

for col in continuous_cols:
    missing_group = df[df['Credit_Product_Missing'] == 1][col]
    non_missing_group = df[df['Credit_Product_Missing'] == 0][col]
    _, p = stats.ttest_ind(missing_group.dropna(), non_missing_group.dropna())
    if p < 0.05:
        significant_associations.append(col)

if significant_associations:
    print("\n初步结论: 存在与缺失情况显著相关的变量，数据可能并非完全随机缺失(MCAR)")
    print(f"显著相关变量: {', '.join(significant_associations)}")
else:
    print("\n初步结论: 未发现与缺失情况显著相关的变量，数据可能是完全随机缺失(MCAR)")