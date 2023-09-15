import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects import conversion
from rpy2.robjects import pandas2ri


def run_flexcode_validation():
    """
    for validation：需要在函数中定义数据输入路径和输出路径，可能容易形成覆盖
    @return: 指定路径的 cde
    """
    # 定义R代码字符串
    r_code = """
    library(readxl)
    library(FlexCoDE)
    library(writexl)

    N <- 200
    path <- paste0("C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/",N)

    for (i in 0:4) {
      df <- read_excel(paste0(path, "data", i, ".xlsx"), sheet = "train")
      df_test <- read_excel(paste0(path, "data", i, ".xlsx"), sheet = "validation")

      set.seed(1)
      sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))

      data_train  <- df[sample, ]
      ntrain = nrow(data_train)
      xtrain = data_train[1:ntrain,1:3]
      ztrain = data_train[1:ntrain,4]

      data_validation   <- df[!sample, ]
      nvalidation = nrow(data_validation)
      xvalidation = data_validation[1:nvalidation,1:3]
      zvalidation = data_validation[1:nvalidation,4]

      data_test <- df_test
      ntest = nrow(data_test)
      xtest = data_test[1:ntest,1:3]
      ztest = data_test[1:ntest,4]

      # conditional density estimation caculation
      fit = fitFlexCoDE(xtrain,ztrain,xvalidation,zvalidation,xtest,ztest,
                        nIMax = 10,
                        regressionFunction = regressionFunction.NW,
                        n_grid = 1000)
      predictedValues = predict(fit,xtest,B=1000)  # B的大小决定cde的稀疏
      cde = as.data.frame(predictedValues$CDE)
      grid = as.data.frame(predictedValues$z)
      names(grid) = c('a')

      # par(mfrow=c(2,2))
      # # par(mar=c(1,1,1,1))
      # for (col in 1:4){
      #   plot(predictedValues$z,predictedValues$CDE[col,],col='lightblue')  # z_grid, cde
      #   loc = as.numeric(4*xtest[col,1]+2*xtest[col,2]+xtest[col,3])   # A = W * X
      #   lines(predictedValues$z,dnorm(predictedValues$z,loc,1),col='red')  #真实cd
      # } 

      output_list = list(cde,grid)
      write_xlsx(output_list, path = paste0(path, "CDE", i, ".xlsx"))
    }
    """
    # 执行R代码
    robjects.r(r_code)
    print("CDE calculation for validation completed")

#     path <- {path}


def run_flexcode_test(sample_num):
    """
    for test：需要在函数中定义数据输入路径和输出路径，可能容易形成覆盖
    @return: 指定路径的 cde
    """
    # 定义R代码字符串
    r_code = f"""
    library(readxl)
    library(FlexCoDE)
    library(writexl)

    N <- {sample_num}
    path <- paste0("C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/",N)
    
    df <- read_excel(paste0(path, "data.xlsx"), sheet = "train")
    df_test <- read_excel(paste0(path, "data.xlsx"), sheet = "test")
    
    set.seed(1)
    sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
    
    data_train  <- df[sample, ]
    ntrain = nrow(data_train)
    xtrain = data_train[1:ntrain,1:3]
    ztrain = data_train[1:ntrain,4]
    
    data_validation   <- df[!sample, ]
    nvalidation = nrow(data_validation)
    xvalidation = data_validation[1:nvalidation,1:3]
    zvalidation = data_validation[1:nvalidation,4]
    
    data_test <- df_test
    ntest = nrow(data_test)
    xtest = data_test[1:ntest,1:3]
    ztest = data_test[1:ntest,4]
    
    # conditional density estimation caculation
    fit = fitFlexCoDE(xtrain,ztrain,xvalidation,zvalidation,xtest,ztest,
                    nIMax = 10,
                    regressionFunction = regressionFunction.NW,
                    n_grid = 1000)
    predictedValues = predict(fit,xtest,B=1000)  # B的大小决定cde的稀疏
    cde = as.data.frame(predictedValues$CDE)
    grid = as.data.frame(predictedValues$z)
    names(grid) = c('a')
    
    output_list = list(cde,grid)
    write_xlsx(output_list, path = paste0(path, "CDE.xlsx"))
    """
    # 执行R代码
    robjects.r(r_code)
    print("CDE calculation for test completed")


def run_flexcode_empirical(sample_num):
    """
    @param sample_num:
    @return: simulation_empirical 路径中
    """
    # 定义R代码字符串
    r_code = f"""
    library(readxl)
    library(FlexCoDE)
    library(writexl)

    N <- {sample_num}
    # path <- paste0("C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/simulation_empirical/",N)
    path <- paste0("C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/test/",N)

    df <- read_excel(paste0(path, "data.xlsx"), sheet = "train")
    df_test <- read_excel(paste0(path, "data.xlsx"), sheet = "test")

    set.seed(1)
    sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))

    data_train  <- df[sample, ]
    ntrain = nrow(data_train)
    xtrain = data_train[1:ntrain,1:3]
    ztrain = data_train[1:ntrain,4]

    data_validation   <- df[!sample, ]
    nvalidation = nrow(data_validation)
    xvalidation = data_validation[1:nvalidation,1:3]
    zvalidation = data_validation[1:nvalidation,4]

    data_test <- df_test
    ntest = nrow(data_test)
    xtest = data_test[1:ntest,1:3]
    ztest = data_test[1:ntest,4]

    # conditional density estimation caculation
    fit = fitFlexCoDE(xtrain,ztrain,xvalidation,zvalidation,xtest,ztest,
                    nIMax = 10,
                    regressionFunction = regressionFunction.NW,
                    n_grid = 1000)
    predictedValues = predict(fit,xtest,B=1000)  # B的大小决定cde的稀疏
    cde = as.data.frame(predictedValues$CDE)
    grid = as.data.frame(predictedValues$z)
    names(grid) = c('a')

    output_list = list(cde,grid)
    write_xlsx(output_list, path = paste0(path, "CDE.xlsx"))
    """
    # 执行R代码
    robjects.r(r_code)
    print("CDE calculation for test completed")



# if __name__ == "__main__":
#     run_flexcode_validation()


# pandas2ri.activate()
# # 定义R代码字符串
# r_code = """
# cde_calculation <- function(df, df_test) {
# set.seed(1)
# sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
#
# data_train <- df[sample, ]
# ntrain <- nrow(data_train)
# xtrain <- data_train[1:ntrain, 1:3]
# ztrain <- data_train[1:ntrain, 4]
#
# data_validation <- df[!sample, ]
# nvalidation <- nrow(data_validation)
# xvalidation <- data_validation[1:nvalidation, 1:3]
# zvalidation <- data_validation[1:nvalidation, 4]
#
# data_test <- df_test
# ntest <- nrow(data_test)
# xtest <- data_test[1:ntest, 1:3]
# ztest <- data_test[1:ntest, 4]
#
# fit <- fitFlexCoDE(xtrain, ztrain, xvalidation, zvalidation, xtest, ztest,
#                 nIMax = 10,
#                 regressionFunction = regressionFunction.NW,
#                 n_grid = 1000)
# predictedValues <- predict(fit, xtest, B=1000)
# cde <- as.data.frame(predictedValues$CDE)
# grid <- as.data.frame(predictedValues$z)
# names(grid) <- c('a')
#
# return(list(cde = cde, grid = grid))
# }
# """
#
# # Load required R packages
# robjects.r['library']('FlexCoDE')
# base = importr('base')
# FlexCoDE = importr('FlexCoDE')
#
# df = pd.read_excel('C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/1000data.xlsx', sheet_name='train')
# df_test = pd.read_excel('C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/1000data.xlsx', sheet_name='test')
# # cde, grid = run_flexcode(df, df_test)
#
# # Convert pandas DataFrames to R data frames
# df_r = pandas2ri.py2ri(df)
# df_test_r = pandas2ri.py2ri(df_test)
#
# # Create an R function using the R code
# r_function = robjects.r(r_code)
#
# # Call the R function with your dataframes
# result = r_function(df_r, df_test_r)
#
# # Convert the R results to pandas DataFrames
# cde_result = pandas2ri.ri2py(result.rx2('cde'))
# grid_result = pandas2ri.ri2py(result.rx2('grid'))


# # Create an R function using the R code
# r_function = robjects.r(r_code)
#
# # Call the R function with your dataframes
# df_r = robjects.conversion.py2rpy(df)
# df_test_r = robjects.conversion.py2rpy(df_test)
# result = r_function(df_r, df_test_r)
#
# # Convert the R results to Python data structures
# cde_result = robjects.conversion.rpy2py(result.rx2('cde'))
# grid_result = robjects.conversion.rpy2py(result.rx2('grid'))


# r_pkg = SignatureTranslatedAnonymousPackage(r_code, "r_pkg")
# # Load the R package into the R environment
# robjects.r['library'](r_pkg)
#
# # Call the R function with your dataframes
# result = robjects.r['my_function'](df, df_test)
#
# # Convert the R results to Python data structures
# cde_result = result.rx2('cde')
# grid_result = result.rx2('grid')

# def run_flexcode(df, df_test):
#     """
#     @param df:
#     @param df_test:
#     @return: cde, grid
#     """
#     # 转换输入数据为 R 的数据结构
#     r_df = robjects.DataFrame(df)
#     r_df_test = robjects.DataFrame(df_test)
#
#     # 执行 R 代码
#     robjects.r(r_code_template)
#
#     # 获取结果
#     cde = robjects.globalenv['cde']
#     grid = robjects.globalenv['grid']


# def run_flexcode(df, df_test):
#     """
#     @param df:
#     @param df_test:
#     @return: cde, grid
#     """
#     # 定义R代码字符串
#     r_code_template = """
#     library(FlexCoDE)
#
#     set.seed(1)
#     sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
#
#     data_train <- df[sample, ]
#     ntrain <- nrow(data_train)
#     xtrain <- data_train[1:ntrain, 1:3]
#     ztrain <- data_train[1:ntrain, 4]
#
#     data_validation <- df[!sample, ]
#     nvalidation <- nrow(data_validation)
#     xvalidation <- data_validation[1:nvalidation, 1:3]
#     zvalidation <- data_validation[1:nvalidation, 4]
#
#     data_test <- df_test
#     ntest <- nrow(data_test)
#     xtest <- data_test[1:ntest, 1:3]
#     ztest <- data_test[1:ntest, 4]
#
#     fit <- fitFlexCoDE(xtrain, ztrain, xvalidation, zvalidation, xtest, ztest,
#                     nIMax = 10,
#                     regressionFunction = regressionFunction.NW,
#                     n_grid = 1000)
#     predictedValues <- predict(fit, xtest, B=1000)
#     cde <- as.data.frame(predictedValues$CDE)
#     grid <- as.data.frame(predictedValues$z)
#     names(grid) <- c('a')
#     """
#
#     # 将 Python DataFrame 转换为 R 数据结构
#     pandas2ri.activate()
#     robjects.globalenv['df'] = pandas2ri.py2ri(df)
#     robjects.globalenv['df_test'] = pandas2ri.py2ri(df_test)
#
#     # 格式化R代码字符串并执行
#     formatted_r_code = r_code_template
#     robjects.r(formatted_r_code)
#
#     # 从 R 中获取结果并转换为 Pandas DataFrame
#     cde = pandas2ri.ri2py(robjects.globalenv['cde'])
#     grid = pandas2ri.ri2py(robjects.globalenv['grid'])
#
#     return cde, grid


# def run_flexcode(df, df_test):
#     """
#     @param df:
#     @param df_test:
#     @return: cde, grid
#     """
#     base = importr('base')  # 导入 base 包
#     data_frame = base.data_frame  # 获取数据框函数
#
#     # 将 Python DataFrame 转换为 R 数据框
#     r_df = robjects.DataFrame(df)
#     r_df_test = robjects.DataFrame(df_test)
#     # r_df = data_frame(df)
#     # r_df_test = data_frame(df_test)
#
#     # 将 Python DataFrame 转换为 R 数据结构
#     # r_df = conversion.py2ri(df)
#     # r_df_test = conversion.py2ri(df_test)
#     # pandas2ri.activate()
#     # r_df = pandas2ri.py2ri(df)
#     # r_df_test = pandas2ri.py2ri(df_test)
#
#     # 定义R代码字符串
#     r_code_template = """
#     library(FlexCoDE)
#
#     set.seed(1)
#     sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
#
#     data_train <- df[sample, ]
#     ntrain <- nrow(data_train)
#     xtrain <- data_train[1:ntrain, 1:3]
#     ztrain <- data_train[1:ntrain, 4]
#
#     data_validation <- df[!sample, ]
#     nvalidation <- nrow(data_validation)
#     xvalidation <- data_validation[1:nvalidation, 1:3]
#     zvalidation <- data_validation[1:nvalidation, 4]
#
#     data_test <- df_test
#     ntest <- nrow(data_test)
#     xtest <- data_test[1:ntest, 1:3]
#     ztest <- data_test[1:ntest, 4]
#
#     fit <- fitFlexCoDE(xtrain, ztrain, xvalidation, zvalidation, xtest, ztest,
#                     nIMax = 10,
#                     regressionFunction = regressionFunction.NW,
#                     n_grid = 1000)
#     predictedValues <- predict(fit, xtest, B=1000)
#     cde <- as.data.frame(predictedValues$CDE)
#     grid <- as.data.frame(predictedValues$z)
#     names(grid) <- c('a')
#     """
#
#     # 格式化R代码字符串并执行
#     formatted_r_code = r_code_template
#     robjects.r(formatted_r_code)
#
#     # 获取结果
#     cde = robjects.globalenv['cde']
#     grid = robjects.globalenv['grid']
#
#     return cde, grid


# def run_flexcode(df, df_test):
#     """
#     @param df:
#     @param df_test:
#     @return: cde, grid
#     """
#     # 定义R代码字符串
#     r_code = """
#     library(readxl)
#     library(FlexCoDE)
#     library(writexl)
#
#     # N <- 1000
#     # path <- paste0("C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/",N)
#     #
#     # df <- read_excel(paste0(path, "data.xlsx"), sheet = "train")       # data input
#     # df_test <- read_excel(paste0(path, "data.xlsx"), sheet = "test")
#
#     set.seed(1)
#     sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
#
#     data_train  <- df[sample, ]
#     ntrain = nrow(data_train)
#     xtrain = data_train[1:ntrain,1:3]
#     ztrain = data_train[1:ntrain,4]
#
#     data_validation   <- df[!sample, ]
#     nvalidation = nrow(data_validation)
#     xvalidation = data_validation[1:nvalidation,1:3]
#     zvalidation = data_validation[1:nvalidation,4]
#
#     data_test <- df_test
#     ntest = nrow(data_test)
#     xtest = data_test[1:ntest,1:3]
#     ztest = data_test[1:ntest,4]
#
#     # conditional density estimation caculation
#     fit = fitFlexCoDE(xtrain,ztrain,xvalidation,zvalidation,xtest,ztest,
#                     nIMax = 10,
#                     regressionFunction = regressionFunction.NW,
#                     n_grid = 1000)
#     predictedValues = predict(fit,xtest,B=1000)  # B的大小决定cde的稀疏
#     cde = as.data.frame(predictedValues$CDE)
#     grid = as.data.frame(predictedValues$z)
#     names(grid) = c('a')
#
#     # par(mfrow=c(2,2))
#     # # par(mar=c(1,1,1,1))
#     # for (col in 1:4){
#     #   plot(predictedValues$z,predictedValues$CDE[col,],col='lightblue')  # z_grid, cde
#     #   loc = as.numeric(4*xtest[col,1]+2*xtest[col,2]+xtest[col,3])   # A = W * X
#     #   lines(predictedValues$z,dnorm(predictedValues$z,loc,1),col='red')  #真实cd
#     # }
#
#     # output_list = list(cde,grid)
#     # write_xlsx(output_list, path = paste0(path, "CDE", i, ".xlsx"))    # output path
#     # }
#     """
#     # 执行R代码
#     robjects.r(r_code)
#     return cde, grid


# r_code = """
# library(readxl)
# library(FlexCoDE)
# library(writexl)
#
# N <- 1000
# path <- paste0("C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/",N)
#
# for (i in 0:4) {
#   df <- read_excel(paste0(path, "data", i, ".xlsx"), sheet = "train")       # data input
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
#   write_xlsx(output_list, path = paste0(path, "CDE", i, ".xlsx"))    # output path
# }
# """