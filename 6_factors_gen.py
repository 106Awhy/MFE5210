import pandas as pd
import numpy as np

# 1. 读取原始数据

data = pd.read_pickle('data1_normal.pkl')


def standardize_series_cross_sectionally(series_to_standardize, date_level_name='date'):
    """对给定的Series在每个时间截面进行Z-Score标准化。"""
    return series_to_standardize.groupby(level=date_level_name).transform(lambda x: (x - x.mean()) / x.std(ddof=1))

# 2. 计算6个新的复合因子
factor_data_new = pd.DataFrame(index=data.index)

# --- 因子 1: 复合价值因子 (COMP_VALUE) ---

raw_value_factors = [
    'book_value_to_total_mktcap_mrq',
    'revenue_to_total_mktcap_ttm',
    'net_cash_flow_to_total_mktcap_ttm',
    'book_value_plus_rdexp_to_total_mktcap_ttm'
]
standardized_value_components = []
for factor_name in raw_value_factors:

    standardized_value_components.append(standardize_series_cross_sectionally(data[factor_name]))

factor_data_new['COMP_VALUE'] = sum(standardized_value_components) / len(standardized_value_components)

# --- 因子 2: 复合动量因子 (COMP_MOMENTUM) ---

raw_momentum_factors = [
    'idios_momentum_ff3_252_21',
    'idios_momentum_capm_252_21',
    'tpi_21' 
]
standardized_momentum_components = []
for factor_name in raw_momentum_factors:
    standardized_momentum_components.append(standardize_series_cross_sectionally(data[factor_name]))

factor_data_new['COMP_MOMENTUM'] = sum(standardized_momentum_components) / len(standardized_momentum_components)

# --- 因子 3: 复合质量/盈利能力因子 (COMP_QUALITY_PROFIT) ---

raw_quality_profit_factors = [
    'fscore',
    'gross_profit_to_asset_yoy_chg',
    'net_profit_to_asset_yoy_chg_acc',
    'asset_turnover_yoy_chg'
]
standardized_quality_profit_components = []
for factor_name in raw_quality_profit_factors:
    standardized_quality_profit_components.append(standardize_series_cross_sectionally(data[factor_name]))

factor_data_new['COMP_QUALITY_PROFIT'] = sum(standardized_quality_profit_components) / len(standardized_quality_profit_components)

# --- 因子 4: 复合成长因子 (COMP_GROWTH) ---

raw_growth_factors = [
    'net_profit_yoy_pct_chg_acc',
    'revenue_to_net_operating_asset_yoy_chg'
]
standardized_growth_components = []
for factor_name in raw_growth_factors:
    standardized_growth_components.append(standardize_series_cross_sectionally(data[factor_name]))

factor_data_new['COMP_GROWTH'] = sum(standardized_growth_components) / len(standardized_growth_components)

# --- 因子 5: 复合波动超额因子 (COMP_VOLATILITY_PREMIUM) ---

raw_volatility_factors = [
    'std_unexpected_net_profit_with_drift',
    'std_unexpected_revenue_with_drift'
]
standardized_volatility_components = []
for factor_name in raw_volatility_factors:
    standardized_volatility_components.append(standardize_series_cross_sectionally(data[factor_name]))

factor_data_new['COMP_VOLATILITY_PREMIUM'] = sum(standardized_volatility_components) / len(standardized_volatility_components)

# --- 因子 6: 复合费用控制因子 (COMP_EXPENSE_CONTROL) ---

# 1. adminexp_to_total_mktcap_ttm (管理费用/市值)
# 2. sellexp_to_total_mktcap_ttm (销售费用/市值)

# 方向调整并标准化
adj_admin_exp = data['adminexp_to_total_mktcap_ttm'] * -1.0
z_adj_admin_exp = standardize_series_cross_sectionally(adj_admin_exp)

adj_sell_exp = data['sellexp_to_total_mktcap_ttm'] * -1.0
z_adj_sell_exp = standardize_series_cross_sectionally(adj_sell_exp)

factor_data_new['COMP_EXPENSE_CONTROL'] = (z_adj_admin_exp + z_adj_sell_exp) / 2.0


# 3. 合并 return 列和6个新复合因子
if 'return' in data.columns:
    result_new = pd.concat([data['return'], factor_data_new], axis=1)
    print("已合并 'return' 列。")
else:

    print("警告: 'return' 列未在 'data1_normal.pkl' 中找到。新的因子数据已生成但未包含收益率。")
    result_new = factor_data_new # 或者您可以选择在此处引发错误或进行其他处理

# 4. 保存为新文件，保持MultiIndex格式
new_filename = 'data_6_composite_factors.pkl' # 新的文件名
result_new.to_pickle(new_filename)
print(f"已生成 {new_filename}，包含（或不含）return 和6个新的复合因子，索引格式未变。")