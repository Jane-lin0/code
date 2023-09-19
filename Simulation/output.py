import pandas as pd
from tabulate import tabulate
import inspect
import numpy as np


def equal_space(length, indices_num):
    """
    @param length:
    @param indices_num:
    @return:
    """
    # 计算间距
    indices_step = (length - 1) // (indices_num - 1)

    # 确定等间距的索引
    indices = np.arange(0, length, indices_step)

    # 确保起始索引和最后一个索引被包含
    if indices[0] != 0:
        indices[0] = 0
    if indices[-1] != length - 1:
        indices[-1] = length - 1
    return indices


def subset_index(shape, row_num, col_num):
    """
    随机抽取row_num行col_num列，返回一个对应索引
    """
    # 等间距行索引
    row_indices = equal_space(length=shape[0], indices_num=row_num)

    # 等间距列索引
    col_indices = equal_space(length=shape[1], indices_num=col_num)

    return row_indices, col_indices
# 由于 index 在画图时还有用，因此不和 subset 函数写在一起


def treatment_subset_index(shape, row_list, col_num):
    """
    废弃
    随机抽取row_num行col_num列，返回一个对应索引
    """
    # 获取 treatment_grid_eval 的 index
    row_indices = 0
    # 等间距行索引
    # row_indices = equal_space(length=shape[0], indices_num=row_num)

    # 等间距列索引 
    col_indices = equal_space(length=shape[1], indices_num=col_num)

    return row_indices, col_indices


def subset(ndarray, row_index, col_index):
    """
    从ndarray中提取抽样出来的行和列，并将其存储在一个新的ndarray中返回
    """
    ndarray1 = ndarray[row_index][:, col_index]
    return ndarray1


def print_latex(matrix_output):
    """
    @param matrix_output: ndarray
    @return: matrix_output 的 latex 格式
    """
    table_output = tabulate(matrix_output, tablefmt="latex", floatfmt=".4f")  # 输出保留4位小数

    # 获取矩阵的变量名
    frame = inspect.currentframe().f_back
    var_name = [var_name for var_name, var_val in frame.f_locals.items() if np.array_equal(var_val, matrix_output)]
    if not var_name:
        raise ValueError("Variable name not found.")
    matrix_name = var_name[0]

    # 构造新变量名
    new_name = f"table_{matrix_name}"

    # 在当前命名空间中创建新变量
    globals()[new_name] = table_output

    print("=" * 100, "\n", f"{new_name}:\n {table_output}", "\n")

    return globals()[new_name]


def mean_std_calculation(df):
    coulumn_names = df.columns
    mean_values = df.mean().tolist()
    std_values = df.std().tolist()

    summary_df = pd.DataFrame({
        # 'Column': coulumn_names,
        'Mean': mean_values,
        'Std': std_values
    }).T
    # df_merged = pd.concat([df, summary_df], axis=0, ignore_index=True)
    df_merged = pd.concat([df, summary_df], axis=0)
    return mean_values, std_values, df_merged


