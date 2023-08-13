
library(readxl)
library(FlexCoDE)
library(writexl)

N <- 1000
path <- paste0("C:/Users/janline/OneDrive - stu.xmu.edu.cn/学校/论文/论文代码/simulation_data/",N)

for (i in 0:4) {
  df <- read_excel(paste0(path, "data", i, ".xlsx"), sheet = "train")
  df_test <- read_excel(paste0(path, "data", i, ".xlsx"), sheet = "test")
  
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
