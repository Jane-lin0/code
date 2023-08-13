# import numpy as np
#
# bandwidth_list = np.logspace(-4, 1, num=20)

# import pandas as pd
# from sklearn.model_selection import KFold
#
# from Simulation.data_generating.data_generate_process import train_validation_split
#
# N = 200
# cv = 5
# path = f"C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/{N}"
# df = pd.read_excel(path+'data.xlsx', sheet_name='train')
# # train_validation_split(df=df_train, cv=cv, save_path=path)  # split to validation and test set
# kf = KFold(n_splits=cv, shuffle=True)  # 随机分割数据，不设置 random_state，避免重复
# i = 0
# for train_index, validation_index in kf.split(df):
#     df_train = df.loc[train_index]
#     df_validation = df.loc[validation_index]
#
#     # df_train, df_test = time_moderate(df_train, df_test)  # 调整时间，避免计算综合 brier score 时报错
#
#     df_train.sort_values(by='o', ascending=True, inplace=True)
#     df_validation.sort_values(by='o', ascending=True, inplace=True)
#     # # 是否要排序？要排序，一是便于后续条件生存函数的估计,二是排序后样本的顺序和treatment的顺序一致，否则 IBS 的计算有误
#
#     # df_train.sort_values(by='a', ascending=True, inplace=True)
#     # df_test.sort_values(by='a', ascending=True, inplace=True)   # 便于对比输出的反事实结果？不需要对比
#
#     # 将数据存储到本地
#     writer = pd.ExcelWriter(path + f"data{i}.xlsx", engine='xlsxwriter')
#     df_train.to_excel(writer, sheet_name='train', index=False)
#     df_validation.to_excel(writer, sheet_name='validation', index=False)
#     # writer.save()
#     writer.close()
#     i += 1

# import numpy as np
# from Simulation.ouput import subset_index
#
# mat_test = np.arange(60).reshape(10, 6)
# time_test = np.arange(20)
# row_index, col_index = subset_index(mat_test.shape, row_num=5, col_num=mat_test.shape[1])
# out_test = time_test[row_index]

# # main.py
# import draft1

# print("This is the main program")
# draft1.some_function()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import KFold
# # from Simulation.data_generating.data_generate_process import data_generate
# from sksurv.linear_model import CoxPHSurvivalAnalysis
# from sksurv.metrics import concordance_index_censored, integrated_brier_score
# from sksurv.metrics import concordance_index_censored
# import rpy2.robjects as robjects
# import rpy2
# import os
# from sklearn.metrics import mean_squared_error
# from scipy.integrate import dblquad
# from scipy.integrate import nquad
# from tabulate import tabulate
# import os
# import time

# import rpy2.robjects as robjects
# robjects.r('''
#         # create a function `f`
#         f <- function(r, verbose=FALSE) {
#             if (verbose) {
#                 cat("I am calling f().\n")
#             }
#             2 * pi * r
#         }
#         # call the function `f` with argument value 3
#         f(3)
#         ''')
# r_f = robjects.r['f']
# res = r_f(2)  # <rpy2.robjects.vectors.FloatVector object at 0x00000171AFEB8300> [RTYPES.REALSXP] R classes: ('numeric',)[12.566371]
# a = res + 1  # 不是单纯的一个数


# # 定义R代码字符串
# r_code = """
# library(readxl)
# library(FlexCoDE)
# library(writexl)
#
# N <- 1000
# path <- paste0("C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/",N)
#
# for (i in 0:4) {
#   df <- read_excel(paste0(path, "data", i, ".xlsx"), sheet = "train")
#   df_test <- read_excel(paste0(path, "data", i, ".xlsx"), sheet = "test")
#
#   set.seed(1)
#   sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
#
#   data_train  <- df[sample, ]
#   ntrain = nrow(data_train)
#   xtrain = data_train[1:ntrain,1:3]
#   ztrain = data_train[1:ntrain,4]
#
#   data_validation   <- df[!sample, ]
#   nvalidation = nrow(data_validation)
#   xvalidation = data_validation[1:nvalidation,1:3]
#   zvalidation = data_validation[1:nvalidation,4]
#
#   data_test <- df_test
#   ntest = nrow(data_test)
#   xtest = data_test[1:ntest,1:3]
#   ztest = data_test[1:ntest,4]
#
#   # conditional density estimation caculation
#   fit = fitFlexCoDE(xtrain,ztrain,xvalidation,zvalidation,xtest,ztest,
#                     nIMax = 10,
#                     regressionFunction = regressionFunction.NW,
#                     n_grid = 1000)
#   predictedValues = predict(fit,xtest,B=1000)  # B的大小决定cde的稀疏
#   cde = as.data.frame(predictedValues$CDE)
#   grid = as.data.frame(predictedValues$z)
#   names(grid) = c('a')
#
#   # par(mfrow=c(2,2))
#   # # par(mar=c(1,1,1,1))
#   # for (col in 1:4){
#   #   plot(predictedValues$z,predictedValues$CDE[col,],col='lightblue')  # z_grid, cde
#   #   loc = as.numeric(4*xtest[col,1]+2*xtest[col,2]+xtest[col,3])   # A = W * X
#   #   lines(predictedValues$z,dnorm(predictedValues$z,loc,1),col='red')  #真实cd
#   # }
#
#   output_list = list(cde,grid)
#   write_xlsx(output_list, path = paste0(path, "CDE", i, ".xlsx"))
# }
# """
# # 执行R代码
# robjects.r(r_code)
# robjects.r(r_code, encoding='latin1')
'''
虽报错，但可行
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd4 in position 0: invalid continuation byte
R[write to console]: randomForest 4.7-1.1
R[write to console]: Type rfNews() to see new features/changes/bug fixes.
'''


'''' 不可行 '''
# import subprocess
# with open('temp_script.R', 'w') as f:
#     f.write(r_code)
#
# # 运行R脚本
# subprocess.run(['Rscript', 'temp_script.R'])

# # 读取R脚本输出文件
# with open('output_file.txt', 'r', encoding='latin1') as f:  # 这里使用'latin1'编码
#     r_output = f.read()
#
# # 打印R脚本的输出
# print(r_output)


''' 以下代码可行 '''
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# # 导入R的stats包
# # stats = importr('stats')
#
# # 创建Python中的数据
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
#
# # 将Python中的数据转换为R中的对象
# robjects.globalenv['x'] = robjects.FloatVector(x)
# robjects.globalenv['y'] = robjects.FloatVector(y)
#
# # 在R中执行线性回归
# lm_model = stats.lm('y ~ x')
#
# # 打印回归结果
# print(lm_model)

# 不可行
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
#
# # 加载R包
# base = importr("base")
#
# # 调用R包中的函数
# result = robjects.r['mean'](robjects.IntVector([1, 2, 3, 4, 5]))
#
# # 将R对象转换为Python对象
# python_result = robjects.conversion.rpy2py(result)
#
# print("Mean:", python_result)



# test_change = np.arange(12).reshape(3, -1)
# table_test = print_latex(test_change)
# # if table_test == table_test_change:
# #     print("True")
#
# # def print_latex(matrix_output):
# #     """
# #     @param matrix_output: ndarray
# #     @return: 将 matrix_output 打印成 latex 格式
# #     """
# #     table_output = tabulate(matrix_output, tablefmt="latex", floatfmt=".4f")    # 输出保留4位小数
# #     matrix_name = f"{matrix_output}"  # 有误，需要的是 matrix_output 的变量名，但赋值的是变量 matrix_output，
# #     new_name = f"table_{matrix_name}"
# #     exec(f"{new_name} = '{table_output}' ")
# #     print("=" * 100, f"{new_name}:\n {table_output}", "\n")
# #     # print("=" * 100,table_output, "\n")       # 打印结果，复制粘贴到latex
# #     return new_name

# a_test_for_change = np.arange(12).reshape(3, -1)
# b = a_test_for_change * 2
#
# def find_var_name(obj):
#     """
#     Find the name of a variable that refers to the given object.
#     """
#     for name, val in locals().items():
#         if val is obj:
#             return f"{name}"
#     for name, val in globals().items():
#         if val is obj:
#             return f"{name}"
#
#
# val_name = find_var_name(a_test_for_change)
#
# print(val_name)  # Output: obj


# def find_var_name(obj):
#     """
#     Find the name of a variable that refers to the given object.
#     """
#     for name, val in locals().items():
#         if val is obj:
#             return name
#     for name, val in globals().items():
#         if val is obj:
#             return name
#     return None
#
# a_test_for_change = np.arange(12).reshape(3,-1)
# val_name = find_var_name(a_test_for_change)
#
# print(val_name)


# a = np.arange(10).reshape(2,-1)
# a_name = f" '{a}' "

#
# # 遍历全局变量字典
# for name in globals():
#     # 如果变量与指定变量a相同，则将变量名存储到val_name中
#     if globals()[name] is a:
#         val_name = name
#         break

# 输出变量名和值
# print(val_name)    # 输出：a


# # 将 字符串 "apple" 的变量名从 a 改成 fruit
# a = "apple"
# new_name = "fruit"
# exec(f"{new_name} = '{a}' ", globals())

# a = "apple"
# for i, c in enumerate(a):
#     new_name = f"b{i}"
#     exec(f"{new_name} = '{c}'", globals()) # 将变量添加到全局作用域中
#     print(f"{new_name}: {c}")
# print(b0) # 此处访问变量b0，将会输出a


# print("=" * 100, "\n")
# a = 1
# print(a)
# print("=" * 100, "\n")

# # h_list = np.array([0.01       0.01098541 0.01206793 0.01325711 0.01456348 0.01599859, 0.01757511 0.01930698 0.02120951 0.02329952 0.02559548 0.02811769, 0.03088844 0.03393222 0.03727594 0.04094915 0.04498433 0.04941713, 0.05428675 0.05963623 0.06551286 0.07196857 0.07906043 0.08685114, 0.09540955 0.10481131 0.11513954 0.12648552 0.13894955 0.1526418, 0.16768329 0.184207   0.20235896 0.22229965 0.24420531 0.26826958, 0.29470517 0.32374575 0.35564803 0.39069399 0.42919343 0.47148664, 0.51794747 0.5689866  0.62505519 0.68664885 0.75431201 0.82864277, 0.91029818 1.        ])
# h_list = [0.01       0.01098541 0.01206793 0.01325711 0.01456348 0.01599859, 0.01757511 0.01930698 0.02120951 0.02329952 0.02559548 0.02811769, 0.03088844 0.03393222 0.03727594 0.04094915 0.04498433 0.04941713, 0.05428675 0.05963623 0.06551286 0.07196857 0.07906043 0.08685114, 0.09540955 0.10481131 0.11513954 0.12648552 0.13894955 0.1526418, 0.16768329 0.184207   0.20235896 0.22229965 0.24420531 0.26826958, 0.29470517 0.32374575 0.35564803 0.39069399 0.42919343 0.47148664, 0.51794747 0.5689866  0.62505519 0.68664885 0.75431201 0.82864277, 0.91029818 1.        ]
#
# num_h = len(h_list)

# start_time = time.time()
# for i in range(10):
#     h_list = [1, 0.75, 0.5, 0.25]
#     h_list1 = [1, 0.75, 0.5, 0.25]
#     h_list2 = [1, 0.75, 0.5, 0.25]
#     arr = np.array([h_list, h_list1, h_list2])
#     end_time = time.time()
#     run_time = end_time - start_time
#     print("程序运行时间为：", run_time, "秒")


# h = 0.25
# # 在这里绘制您的图像
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# plt.xlabel('time')
# plt.ylabel('survival probability')
#
# # 将图像保存到本地
# desktop = os.path.expanduser("~/Desktop")
# filename = f"h{int(100*h)}.png"
# filepath = os.path.join(desktop, filename)
# plt.savefig(filepath)

# summary_median_survival = np.empty(shape=(0, 6))
# arr = np.arange(6)
# summary_median_survival = np.vstack((summary_median_survival, arr))
# arr1 = np.random.randint(1,10,6)
# summary_median_survival = np.vstack((summary_median_survival, arr1))

# h = 0.25
# # table_pred = f"table_counterfactual_survival_{int(h * 100)}"
# arr = np.arange(12).reshape(3, 4)
# row_index = np.array([0, 2])
# aa = arr.T[row_index]
# df = pd.DataFrame(arr, index=['h_opt','mse','imse'])
# table = tabulate(df, tablefmt="latex", floatfmt=".4f")
# table_pred = f"\\label{int(h*100)}\n{table}\n"
# table_pred1 = f"\\label{int(h*100)}\n{table}\n"
# print(table_pred, "-"*100, table_pred1, "="*100)

# table_out = f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{My table caption}}\n\\label{{table_counterfactual_survival_{int(h * 100)}}}\n{table}\n\\end{{table}}"
# f"table_out_{h}" == table_out
# a = np.arange(0, 10, 3)

# # 创建一个200x500的随机数组
# data = np.random.rand(20, 50)
#
# # 设置表头和表尾
# header = ['Column {}'.format(i+1) for i in range(data.shape[1])]
# footer = [''] * data.shape[1]
#
# # 将数组转换为包含LaTeX表格的字符串
# table = tabulate(data, headers=header, showindex=False, tablefmt="latex")
#
# # 在字符串中添加sidewaystable环境的开始和结束标记
# table = '\\begin{sidewaystable}\\centering\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{%s}\n%s\\end{tabular}}\n\\end{sidewaystable}' % ('|' + '|'.join(['c']*data.shape[1]) + '|', table)
#
# # 输出LaTeX表格
# print(table)


# # 创建一个200x500的随机数组
# data = np.random.rand(200, 50)
#
# # 设置表头和表尾
# header = ['Column {}'.format(i+1) for i in range(data.shape[1])]
# footer = [''] * data.shape[1]
#
# # 将数组转换为包含LaTeX表格的字符串
# table = tabulate(data, headers=header, showindex=False, tablefmt='latex')
#
# # 在字符串中添加tabularx的开始和结束标记  # 自动调整但没调整成功
# table = '\\begin{table}\\centering\\begin{tabularx}{\\textwidth}{%s}\n%s\\end{tabularx}\\end{table}' % ('|' + '|'.join(['X']*data.shape[1]) + '|', table)
#
# # 输出LaTeX表格
# print(table)


# # 创建一个200x500的随机数组  带标题
# data = np.random.rand(20, 50)
#
# # 设置表头和表尾
# header = ['Column {}'.format(i+1) for i in range(data.shape[1])]
# footer = [''] * data.shape[1]
#
# # 将数组转换为包含LaTeX表格的字符串
# table = tabulate(data, headers=header, showindex=False, tablefmt='latex')
#
# # 在字符串中添加longtable的开始和结束标记
# table = '\\begin{longtable}{%s}\n%s\\end{longtable}' % ('|' + '|'.join(['c']*data.shape[1]) + '|', table)

# 输出LaTeX表格
# print(table)


# # 创建一个200x500的随机数组  # A4 页面显示不出来
# data = np.random.rand(10, 20)

# 将数组转换为包含LaTeX表格的字符串
# table = tabulate(np.concatenate((data[-5:, ])), tablefmt="latex")
#
# # 输出LaTeX表格
# print(table)


# h = 1000**(-1/5)
#
# h3 = 3*h
#
# h4 = 4*h

# survival_est = np.arange(12).reshape(3, 4)
# empty_est = np.zeros(shape=survival_est.shape)

# survival_est = np.arange(10).reshape(2, -1)
# survival_true = np.arange(10).reshape(2, -1)
# grid = np.arange(5)
# # grid = np.tile(grid.flatten(), len(survival_est))
# # survival_est = survival_est.flatten()
# # survival_true = survival_true.flatten()
# #
# # term1 = np.trapz(survival_est**2, grid)
# # term2 = np.trapz(survival_est * survival_true, grid)
# # mise = term1 - 2 * term2
# m = integrated_mean_squared_error(survival_est, survival_true, grid)

# grid = np.arange(5)
# grid = np.tile(grid.flatten(), 3)

# a = np.arange(5)
# b = np.tile(a, 3)
# b = a.item()
# .reshape(2,-1)
# b = a.flatten()

# # 定义被积函数
# def func(x, y):
#     return np.exp(-x*y)
#
# # 定义积分区间
# x_range = [0, 1]
# y_range = [lambda x: 0, lambda x: 1 - x]
#
# # 计算积分
# result, error = nquad(func, [x_range, y_range])
#
# print("结果:", result)
# print("误差:", error)

# # 定义被积函数
# def func(x, y):
#     return x+y
#
# # 定义积分区间和积分区域
# a, b = 0, 1
# low_fun = lambda x: 0
# upper_fun = lambda x: 1 - x
#
# # 计算积分
# result, error = dblquad(func, a, b, low_fun, upper_fun)
#
# print("结果:", result)
# print("误差:", error)


# a=np.trapz([1,2,3], x=[4,6,8])

# y_true = [[0.5, 1],[-1, 1],[7, -6]]
# y_pred = [[0, 2],[-1, 2],[8, -5]]
# mse = mean_squared_error(y_true, y_pred) # 0.708
# MSE =((0.5 - 0)**2 + (1 - 2)**2 + (-1 + 1)**2 + (-1 + 2)**2 + (7 - 8)**2 + (-6 + 5)**2)/6  # 0.70833


# os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.0'
#
#
# # rpy2.rinterface.set_R_HOME('/path/to/R')
#
# print(rpy2.__version__) # 3.5.11
# import rpy2.situation as rpy2situation
# # print(rpy2situation.get_r_version())
# # import rpy2.situation as sit
# #
# # print(sit.get_r_home())
# # print(sit.get_rversion())
#
#
# # os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.0'  # 根据实际安装路径进行修改
# # # 执行一条R语言命令
# result = robjects.r('paste("Hello", "world!")')
# print(result[0])

# N = 1000
# path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# df_train = pd.read_excel(path+"data.xlsx", sheet_name='train')
# df_test = pd.read_excel(path+"data.xlsx", sheet_name='test')
# # df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
# #
# # a_median = np.median(df_test['a'])
# lambda_index = df_test['lambda'][0]
#
# T = df['o']
# treatment_col = df['a']
# num_samples = len(df)
# C = np.random.exponential(scale=np.mean(T) + np.std(T), size=num_samples)     # 删失时间服从指数分布
# C1 = np.random.uniform(low=0, high=treatment_col)

# c = np.array([[4], [5], [1]])
# b = c.flatten()

# treatment_col = np.random.binomial(10, 0.5, size=100)
# C = np.random.uniform(low=0, high=treatment_col)

# a = np.logspace(0.01, 1, 10)
# b = np.logspace(-2, 0, num=10)  # 带宽取值范围在0.01到1之间
# estimator = CoxPHSurvivalAnalysis()
# estimator.fit(data_x_numeric, data_y)
# prediction = estimator.predict(data_x_numeric)
# result = concordance_index_censored(data_y["Status"], data_y["Survival_in_days"], prediction)
# c_index = result[0]

# lis = [1,2,3,4,5]
# a = np.mean(lis)

# cv = 5
# for i in range(cv):
#     print(i+1)

# train_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# train_event = np.array([1, 1, 0, 1, 0])
# test_time = np.array([6.0, 7.0, 8.0])
# # 检查测试集时间是否超出最大观测时间点
# extends = test_time > np.max(train_time)
# if np.any(extends):
#     # 将超出的时间值替换为最大观测时间点
#     test_time = np.where(extends, np.max(train_time), test_time)  # 直接替换


# N = 1000
# path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')
# df_test = pd.read_excel(path+"data.xlsx",sheet_name='test')
# df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
#
# T = df['o']
# treatment_col = df['a']
# num_samples = len(df)
# C = np.random.exponential(scale=np.mean(T) + np.std(T), size=num_samples)     # 删失时间服从指数分布
# C1 = np.random.uniform(low=0, high=treatment_col)

#
# # 检查测试集时间是否超出最大观测时间点
# extends = df_test['o'] > np.max(df_train['o'])
# if np.any(extends):
#     # 将超出的时间值替换为最大观测时间点
#     test_time = np.where(extends, np.max(df_train['o']), df_test['o'])


#
# kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 随机抽取索引
# # 进行交叉验证
# for train_index, test_index in kf.split(df):
#     df_val = df.loc[test_index]
#     # print(test_index, "="*50)

# my_dict = {'a': 3, 'b': 2, 'c': 1}
# k, min_value = min(my_dict.items(), key=lambda x: x[1])
#
# x = my_dict.keys()
# y = my_dict.values()
#
# plt.figure()
# plt.plot(x, y, marker='o')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# a = np.random.choice(250)

# df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, index=[1,3,5])
# idx = df['col1'].idxmax()
# row_to_drop = df.loc[idx]
# df_temp = row_to_drop.to_frame().T


# df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, index=[1,3,5]).reset_index(drop=True)
# idx = np.argmin(df['col1'])
# row_to_drop = df.loc[idx, :]
# df = df.drop(idx)

# test = np.random.randint(1,10,size=5)
# idx = np.argmin(test)

# time_points = np.arange(1, 1000)

# N = 1000
# path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# df_test = pd.read_excel(path+"data.xlsx",sheet_name='test')
# df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')
#
# # idx_test = np.random.choice(df_test.index, size=1).item()
# # idx_test = np.random.choice(df_test.index, size=1)
# # row_to_train = df_test.loc[idx_test]
#
# print(df_test.describe())
# print(df_train.describe())
# print(type(df_test['o']))
# print(df_test['o'].min())
# print(type(df_test['o'].min()))
# print(df_test['o'].min() < df_test['o'].max())

# treatment_idx = np.random.randint(low=0, high=10, size=3)
# colors = ['r', 'g', 'b']
# for idx, color in zip(treatment_idx, colors):
#     print(idx, color)

# estimator = CoxPHSurvivalAnalysis()
# estimator.fit(data_x_numeric, data_y)
# estimator.score()
#
# # Estimate the survival function
# survival = estimator.predict_survival_function(data[['time']])
#
# # Calculate the integrated Brier score
# ibs = integrated_brier_score(survival, data['status'], data['time'], t_max=100)



# N = 100
# path = f"C:/Users/janline/Desktop/simulation_data/{N}"
# data_generate(N, path)

# path = "C:/Users/janline/Desktop/"
# df_train = pd.read_excel(path+"data.xlsx",sheet_name='train')

# list_t = [1,2,3,4,5,6]
#
# weight = np.empty(shape=(0,6))
# # list_normalization = np.array(list_t) / sum(list_t)
# # weight = np.vstack([weight0,list_normalization])
# weight = np.vstack([weight,list_t])
#
# list_t1 = [1,2,3,4,5,5]
# # list_normalization1 = np.array(list_t1) / sum(list_t1)
# # weight = np.vstack([weight, list_normalization1])
# weight = np.vstack([weight,list_t1])
#
# n = weight.shape[0]
# res = []
# for i in range(n):
#     index = np.argmin(np.abs(weight[i, :] - 3))
#     res.append(weight[i, index])


# arr = np.array(list_t).reshape(2,3)
# arr_T = arr.T
# multiply = arr_T @ arr

# time = 1.02
# treat = 3.766
# conditional_survival_est = conditional_survival_estimated[:, col]  # ndarray:(150,)
# kernel_val = gaussian_kernel(a_approx, treat, h)  # ndarray:(150,1), list 150

# fenzi = 0
# for i in [0]:
#     print(pi[i] , conditional_survival_est[i] , kernel_val[i],
#           pi[i] * conditional_survival_est[i] * kernel_val[i])

# total = pi * conditional_survival_est * kernel_val
# fenzi = np.sum(total)
#
# total1 = pi * kernel_val
# fenmu = np.sum(total1)

# survival_est = np.sum(pi * conditional_survival_est * kernel_val)/np.sum(pi * kernel_val)
# survival_estimates.append(survival_est)

# a = np.array([1,2,3]).reshape(-1,1)
# b = a
# c = a*b
# d = a*a*a
# e = sum(d)

# df_test = pd.read_excel("C:/Users/janline/Desktop/simulation_data/data.xlsx",sheet_name='test')
# a_grid = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet2')
#
#
# n_obs = len(df_test)
# a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]  # 长度和 df_test 一致
# a_approx = np.array([a_grid.loc[i].item() for i in a_approx_index]).reshape(-1,1)
# # a_approx = [a_grid.loc[i].item() for i in a_approx_index]
#
# treatment_grid = np.linspace(min(a_approx), max(a_approx), num=100)  # 连续 treatment 取值网格点
# for treat in treatment_grid:
#     print(treat)

# a = [-1,1,2]+[1]

# df_train = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='train')
# df_validation = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='validation')
# df_test = pd.read_excel("C:/Users/janline/Desktop/data.xlsx",sheet_name='test')
#
# df = pd.concat([df_train, df_validation, df_test], axis=0)
#
# cde_estimates = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet1')
# a_grid = pd.read_excel("C:/Users/janline/Desktop/file_show.xlsx",sheet_name='Sheet2')
#
# n_obs = len(df_test)
# a_approx_index = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]  # 长度和 df_test 一致
# a_approx = np.array([a_grid.loc[i].item() for i in a_approx_index])




#
# n_obs = len(df_test)
# nns = [np.argmin(np.abs(a_grid - df_test['a'][i])) for i in range(n_obs)]
# cde_list =[]
# for x_index,grid_index in enumerate(nns):
#     cde_list.append(cde_estimates.iloc[x_index, grid_index])
#
# # cde_list = [cde_estimates[range(n_obs), nns]]
#
# # num_samples = 1000
# # da = np.random.randn(num_samples).reshape(-1,1)