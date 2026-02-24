'''
In this file we are going to clear all the outliers from all numerical features
'''
import numpy as np
import pandas as pd
import sys
import logging
from log_code import setup_logging
logger = setup_logging('trimming')
import warnings
warnings.filterwarnings('ignore')

def trim_tech(train_num , test_num):
    try:
        for i in train_num.columns:
            iqr = train_num[i].quantile(0.75) - train_num[i].quantile(0.25)
            upper_limit = train_num[i].quantile(0.75) + (1.5 * iqr)
            lower_limit = train_num[i].quantile(0.25) - (1.5 * iqr)
            train_num[i+'_trim'] = np.where(train_num[i] > upper_limit,upper_limit,np.where(train_num[i] < lower_limit,lower_limit,train_num[i]))

            test_num[i+'_trim'] = np.where(test_num[i] > upper_limit, upper_limit,np.where(test_num[i] < lower_limit, lower_limit, test_num[i]))

        logger.info(f'After Trimming column names : {train_num.columns}')
        logger.info(f'After Trimming column names : {test_num.columns}')

        f = []
        for j in train_num.columns:
            if '_trim' not in j and '_days_trim' not in j:
                f.append(j)
        logger.info(f'The Trim and Days_Trim column names : {f}')
        train_num = train_num.drop(f, axis=1)
        test_num = test_num.drop(f, axis=1)

        return train_num, test_num

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')





