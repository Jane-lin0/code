import numpy as np
import pandas as pd

# all_clinical00 = pd.read_sas(r"C:\Users\janline\OneDrive - stu.xmu.edu.cn\桌面\empirical_data\OAIdata\AllClinical_SAS\allclinical00.sas7bdat")
# physical_activity_scale = all_clinical00['V00PASE']

# Mean minutes per day spent in physical activity intensity level for each participant having 4 or more valid monitoring days
# accelerometry06 = pd.read_sas(r"C:\Users\janline\OneDrive - stu.xmu.edu.cn\桌面\empirical_data\OAIdata\Ancillary_SAS\Accelerometry_SAS\accelerometry06.sas7bdat")

# accelerometry08 = pd.read_sas(r"C:\Users\janline\OneDrive - stu.xmu.edu.cn\桌面\empirical_data\OAIdata\Ancillary_SAS\Accelerometry_SAS\accelerometry08.sas7bdat")

# acceldatabyday08 = pd.read_sas(r"C:\Users\janline\OneDrive - stu.xmu.edu.cn\桌面\empirical_data\OAIdata\Ancillary_SAS\Accelerometry_SAS\acceldatabyday08.sas7bdat")

# print(np.sum(physical_activity_scale == physical_activity_scale0))  # 比样本数小

# if allclinical00.equals(all_clinical00):
#     print("两个DataFrame的值相同")
# else:
#     print("不同")
# print(np.array_equal(allclinical00.values, all_clinical00.values)) # 返回 False，可能数据类型不匹配，例如一个整数存储，另一个是浮点数

# print(allclinical00['V00PASE'].describe())  # 0-531

# allclinical11 = pd.read_sas(r"C:\Users\janline\OneDrive - stu.xmu.edu.cn\桌面\empirical_data\OAIdata\CompleteData_SAS\allclinical11.sas7bdat")
# 2015年，195个变量，没有V00PASE

allclinical00 = pd.read_sas(r"C:\Users\janline\OneDrive - stu.xmu.edu.cn\桌面\empirical_data\OAIdata\CompleteData_SAS\allclinical00.sas7bdat")

df = pd.DataFrame()

df["treatment_PASE"] = allclinical00['V00PASE']  # physical_activity_scale，2017年，1187个变量

df["outcome_speed"] = allclinical00['V0020MPACE']  # pace speed

df['event'] = allclinical00['P02KRS3']

# covariance
df["age"] = allclinical00['V00AGE']
df["maried"] = allclinical00['V00MARITST']

keen_pain_right = allclinical00['V00P7RKRCV']
keen_pain_left = allclinical00['V00P7LKRCV']
df['keen_pain'] = pd.concat([keen_pain_right, keen_pain_left], axis=1).max(axis=1)

WOMAC_left = allclinical00['V00WOMTSL']
WOMAC_right = allclinical00['V00WOMTSR']
df['WOMAC'] = pd.concat([WOMAC_left, WOMAC_right], axis=1).max(axis=1)

hip_pain_right = allclinical00['P01HPR12CV']
hip_pain_left = allclinical00['P01HPL12CV']
df['hip_pain'] = pd.concat([hip_pain_left, hip_pain_right], axis=1).max(axis=1)

ankle_pain_right = allclinical00['P01OJPNRA']
ankle_pain_left = allclinical00['P01OJPNLA']
df['ankle_pain'] = pd.concat([ankle_pain_left, ankle_pain_right], axis=1).max(axis=1)

foot_pain_right = allclinical00['P01OJPNRF']
foot_pain_left = allclinical00['P01OJPNLF']
df['foot_pain'] = pd.concat([foot_pain_left, foot_pain_right], axis=1).max(axis=1)

df['smoke'] = allclinical00['V00SMOKER']

df['alcohol'] = allclinical00['V00DRNKAMT']

df['comorbidity'] = allclinical00['V00COMORB']  # 疾病情况

df['depression'] = allclinical00['V00CESD']

df['BMI'] = allclinical00['P01BMI']

# df.to_excel("./Empirical/OAIdata/Data_Osteoarthritis_Initiative.xlsx")

