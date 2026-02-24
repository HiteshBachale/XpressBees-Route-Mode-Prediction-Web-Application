import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import logging
from log_code import setup_logging
logger = setup_logging('hypothesis_testing')
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr

def fs(train_num,test_num,train_dep,test_dep):
    try:
        logger.info(f'Dependent variable unique labels before are : {train_dep.unique()} :')
        train_dep = train_dep.map({'Air':1,'Surface':0}).astype(int)
        logger.info(f'Dependent variable unique labels after are : {train_dep.unique()} :')
        c = []
        for i in train_num.columns:
            c.append(pearsonr(train_num[i] , train_dep))
        c = np.array(c)
        logger.info(f'Hypothesis Testing : {c}')

        '''result = pd.Series(c[:, 1], index=train_num.columns)
        plt.figure(figsize=(10,15))
        result.sort_values(ascending=True).plot.bar()
        plt.show()'''

        train_num = train_num.drop(['ProcessLocation3_BagInScanDate_days_trim'],axis=1)
        test_num = test_num.drop(['ProcessLocation3_BagInScanDate_days_trim'],axis=1)
        logger.info(f'Train Column Names : {train_num.columns}')
        logger.info(f'Test Column Names : {test_num.columns}')
        return train_num,test_num

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
