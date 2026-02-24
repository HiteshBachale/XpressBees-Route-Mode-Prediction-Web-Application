'''
In this file we are going to implement log transformation concept
'''
import numpy as np
import pandas as pd
import sys
import logging
from log_code import setup_logging
logger = setup_logging('transformation')
import warnings
warnings.filterwarnings('ignore')

def log_tran(X_train_num,X_test_num):
    try:
        for i in X_train_num.columns:
            if pd.api.types.is_numeric_dtype(X_train_num[i]):
                # Numeric columns → log transform (safe log1p)
                X_train_num[i + '_log'] = np.log1p(X_train_num[i])
                X_test_num[i + '_log'] = np.log1p(X_test_num[i])

            elif pd.api.types.is_datetime64_any_dtype(X_train_num[i]):
                # Define reference date once, based on training data
                ref_date = X_train_num[i].min()
                X_train_num[i + '_days'] = (X_train_num[i] - ref_date).dt.days
                X_test_num[i + '_days'] = (X_test_num[i] - ref_date).dt.days

        logger.info(f'Log/Date Transformation Completed Successfully : {X_train_num.columns}')

        f = []
        for j in X_train_num.columns:
            if '_log' not in j and '_days' not in j:
                f.append(j)
        logger.info(f'The Log and Days column names : {f}')
        X_train_num = X_train_num.drop(f, axis=1)
        X_test_num = X_test_num.drop(f, axis=1)

        return X_train_num, X_test_num

    except Exception as e:
         er_ty, er_msg, er_lin = sys.exc_info()
         logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
