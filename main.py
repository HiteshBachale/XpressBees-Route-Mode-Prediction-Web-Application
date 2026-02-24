"""
In the Main file we are going to load data and call respected functions for model
development
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import sys
import logging
import pickle
from log_code import setup_logging
logger = setup_logging('main')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from random_sample import random_value_filling
from transformation import log_tran
from trimming import trim_tech
from constant import con
from quasi_constant import con_
from hypothesis_testing import fs
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler # z_score
from train_algo import common
from sklearn.tree import DecisionTreeClassifier
'''from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier'''
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

class XPRESS_BEES:
    try:
        def __init__(self,path):
            self.df = pd.read_excel(path)
            logger.info(f"Data Loaded Successfull : {self.df.shape}")
            '''self.df = self.df.drop([150000,150001],axis=0)'''
            self.df = self.df.drop(['Shipping ID','Origin_HubName_HubCity'],axis=1)
            # Select all columns except the one at index 8 for independent variables
            self.X = self.df.drop(self.df.columns[6], axis=1)  # independent data
            self.y = self.df.iloc[:, 6]  # dependent data
            logger.info(f'X Columns Names : {self.X}')
            logger.info(f'Y Columns Names:{self.y}')
            # checking if the data is clean or not:
            logger.info(f"Missing Values in the data : {self.df.isnull().sum()}")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.2,random_state=42)
    except Exception as e:
        er_ty,er_msg,er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def missing_values(self):
        try:
            # from 5 different technique we finalized random sample imputation technique
            self.X_train,self.X_test = random_value_filling(self.X_train,self.X_test)
            self.X_train = self.X_train.drop(['ProcessLocation3_BagOutScanDate','Destination_BagOutScanDate'],axis=1)
            self.X_test = self.X_test.drop(['ProcessLocation3_BagOutScanDate','Destination_BagOutScanDate'], axis=1)
            logger.info(f'{self.X_train.isnull().sum()}')
            logger.info(f'{self.X_test.isnull().sum()}')
            #logger.info(f'{self.X_train.info()}')
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logger.info(f'Numerical Column Names: {self.X_train_num.columns}')
            logger.info(f'Categorical Column Names: {self.X_train_cat.columns}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def handle_outliers(self):
        try:
            '''for i in self.X_train_num.columns:
                sns.boxplot(x = self.X_train_num[i])
                plt.show()'''
            # Let's convert each Numerical Column into Log Transformation
            # 70 % of outliers we can remove
            # Then we pass to Trimming Formula to remove all outliers in the data
            self.X_train_num,self.X_test_num = log_tran(self.X_train_num,self.X_test_num)
            self.X_train_num,self.X_test_num = trim_tech(self.X_train_num,self.X_test_num)
            logger.info(f'Train Data Features : {self.X_train_num.columns}')
            logger.info(f'Test Data Features : {self.X_test_num.columns}')
            '''for i in self.X_train_num.columns:
                sns.boxplot(x = self.X_train_num[i])
                plt.show()'''

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def feature_selection(self):
        try:
            self.X_train_num, self.X_test_num = con(self.X_train_num, self.X_test_num)
            logger.info(f'Train Column Names : {self.X_train_num.columns}')
            logger.info(f'Test Column Names : {self.X_test_num.columns}')
            self.X_train_num, self.X_test_num = con_(self.X_train_num, self.X_test_num)
            logger.info(f'Train Column Names : {self.X_train_num.columns}')
            logger.info(f'Test Column Names : {self.X_test_num.columns}')
            self.X_train_num,self.X_test_num = fs(self.X_train_num,self.X_test_num,self.y_train,self.y_test)
            logger.info(f'Train Column Names : {self.X_train_num.columns}')
            logger.info(f'Test Column Names : {self.X_test_num.columns}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def cat_to_num(self):
        try:
            logger.info(f'Categorical Column Names')
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            # Applying One-hot encoding in 'lane','OriginHubName','OriginHubCity','OriginHubZoneName','ProcessLocation2_HubName_HubCity',
            # 'ProcessLocation3_HubName_HubCity','ProcessLocation4_HubName_HubCity','Destination_HubName_HubCity','DestinationHubName',
            # 'DestinationHubCity','DestinationHubZoneName' Columns
            oh = OneHotEncoder(categories='auto', drop='first', handle_unknown='ignore')
            oh.fit(self.X_train_cat[['lane','OriginHubName','OriginHubCity','OriginHubZoneName','ProcessLocation2_HubName_HubCity',
            'ProcessLocation3_HubName_HubCity','ProcessLocation4_HubName_HubCity','Destination_HubName_HubCity','DestinationHubName',
            'DestinationHubCity','DestinationHubZoneName']])
            logger.info(f'{oh.categories_}')
            logger.info(f'{oh.get_feature_names_out()}')
            res = oh.transform(self.X_train_cat[['lane','OriginHubName','OriginHubCity','OriginHubZoneName','ProcessLocation2_HubName_HubCity',
            'ProcessLocation3_HubName_HubCity','ProcessLocation4_HubName_HubCity','Destination_HubName_HubCity','DestinationHubName',
            'DestinationHubCity','DestinationHubZoneName']]).toarray()
            res_test = oh.transform(self.X_test_cat[['lane','OriginHubName','OriginHubCity','OriginHubZoneName','ProcessLocation2_HubName_HubCity',
            'ProcessLocation3_HubName_HubCity','ProcessLocation4_HubName_HubCity','Destination_HubName_HubCity','DestinationHubName',
            'DestinationHubCity','DestinationHubZoneName']]).toarray()
            f = pd.DataFrame(res, columns=oh.get_feature_names_out())
            f_test = pd.DataFrame(res_test, columns=oh.get_feature_names_out())
            self.X_train_cat.reset_index(drop=True, inplace=True)
            f.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)
            f_test.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, f], axis=1)
            self.X_test_cat = pd.concat([self.X_test_cat, f_test], axis=1)

            '''logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_train_cat.sample(15)}')
            logger.info(f'{self.X_train_cat.isnull().sum()}')
            logger.info(f'{self.X_test_cat.columns}')
            logger.info(f'{self.X_test_cat.sample(15)}')
            logger.info(f'{self.X_test_cat.isnull().sum()}')'''

            # ShipmentStatus -> We are going to apply Ordinal Encoder Technique
            od = OrdinalEncoder()
            od.fit(self.X_train_cat[['ShipmentStatus']])
            logger.info(f'{od.categories_}')
            logger.info(f'column_names : {od.get_feature_names_out()}')
            res1 = od.transform(self.X_train_cat[['ShipmentStatus']])
            res1_test = od.transform(self.X_test_cat[['ShipmentStatus']])
            c_names = od.get_feature_names_out()
            f1 = pd.DataFrame(res1, columns=c_names+['_con'])
            f1_test = pd.DataFrame(res1_test, columns=c_names+['_con'])
            self.X_train_cat.reset_index(drop=True, inplace=True)
            f1.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)
            f1_test.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, f1], axis=1)
            self.X_test_cat = pd.concat([self.X_test_cat, f1_test], axis=1)

            '''logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_train_cat.sample(15)}')
            logger.info(f'{self.X_train_cat.isnull().sum()}')'''

            self.X_train_cat = self.X_train_cat.drop(['lane','OriginHubName','OriginHubCity','OriginHubZoneName','ProcessLocation2_HubName_HubCity',
            'ProcessLocation3_HubName_HubCity','ProcessLocation4_HubName_HubCity','Destination_HubName_HubCity','DestinationHubName',
            'DestinationHubCity','DestinationHubZoneName','ShipmentStatus'],axis=1)
            self.X_test_cat = self.X_test_cat.drop(['lane','OriginHubName','OriginHubCity','OriginHubZoneName','ProcessLocation2_HubName_HubCity',
            'ProcessLocation3_HubName_HubCity','ProcessLocation4_HubName_HubCity','Destination_HubName_HubCity','DestinationHubName',
            'DestinationHubCity','DestinationHubZoneName','ShipmentStatus'],axis=1)

            logger.info(f'{self.X_train_cat.isnull().sum()}')
            logger.info(f'{self.X_test_cat.isnull().sum()}')

            logger.info(f'{self.X_train_cat.sample(10)}')
            logger.info(f'{self.X_test_cat.sample(10)}')

            logger.info(f'y_train_data : {self.y_train.unique()}')
            logger.info(f'y_train_data : {self.y_train.isnull().sum()}')
            logger.info(f'y_test_data : {self.y_test.unique()}')
            logger.info(f'y_test_data : {self.y_test.isnull().sum()}')

            # dependent varibale should be converted using label encoder
            logger.info(f'{self.y_train[:10]}')
            lb = LabelEncoder()
            lb.fit(self.y_train)
            self.y_train = lb.transform(self.y_train)
            self.y_test = lb.transform(self.y_test)

            logger.info(f'detailed : {lb.classes_} ')
            logger.info(f'{self.y_train[:10]}')
            logger.info(f'y_train_data : {self.y_train.shape}')
            logger.info(f'y_test_data : {self.y_test.shape}')

            # 1 -> Surface
            # 0 -> Air

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def merge_data(self):
        try:
            # reset index so that we can concat data perfectlly
            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)

            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            logger.info(f'Training_data shape : {self.training_data.shape} -> {self.training_data.columns}')
            logger.info(f'Testing_data shape : {self.testing_data.shape} -> {self.testing_data.columns}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def balanced_data(self):
        try:
            logger.info('----------------Before Balancing------------------------')
            logger.info(f'Total row for Surface category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 1)}')
            logger.info(f'Total row for Air category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 0)}')
            logger.info(f'---------------After Balancing-------------------------')
            sm = SMOTE(random_state=42)
            self.training_data_res, self.y_train_res = sm.fit_resample(self.training_data, self.y_train)
            logger.info(f'Total row for Surface category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 1)}')
            logger.info(f'Total row for Air category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 0)}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def feature_scaling(self):
        try:
            logger.info('---------Before scaling-------')
            logger.info(f'{self.training_data_res.head(4)}')
            sc = StandardScaler()
            sc.fit(self.training_data_res)
            self.training_data_res_t = sc.transform(self.training_data_res)
            self.testing_data_t = sc.transform(self.testing_data)
            with open('standard_scalar.pkl','wb') as t:
                pickle.dump(sc,t)
            logger.info('----------After scaling--------')
            logger.info(f'{self.training_data_res_t}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def train_models(self):
        try:
            common(self.training_data_res_t, self.y_train_res, self.testing_data_t, self.y_test)

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def best_model(self):
        try:
            logger.info(f'============ Finalized Model Decision Tree Classifier ===============')

            '''logger.info(f'KNN')
            self.knn_reg = KNeighborsClassifier(n_neighbors=5)
            self.knn_reg.fit(self.training_data_res_t,self.y_train_res)
            logger.info(f'Model Test Accuracy : {accuracy_score(self.y_test, self.knn_reg.predict(self.testing_data))}')
            logger.info(f'Confusion Matrix : {confusion_matrix(self.y_test, self.knn_reg.predict(self.testing_data))}')
            logger.info(f'Classification Report : {classification_report(self.y_test, self.knn_reg.predict(self.testing_data))}')

            logger.info(f'NB')
            self.nb_reg = GaussianNB()
            self.nb_reg.fit(self.training_data_res_t,self.y_train_res)
            logger.info(f'Model Test Accuracy : {accuracy_score(self.y_test, self.nb_reg.predict(self.testing_data))}')
            logger.info(f'Confusion Matrix : {confusion_matrix(self.y_test, self.nb_reg.predict(self.testing_data))}')
            logger.info(f'Classification Report : {classification_report(self.y_test, self.nb_reg.predict(self.testing_data))}')

            logger.info(f'LR')
            self.reg_log = LogisticRegression()
            self.reg_log.fit(self.training_data_res_t,self.y_train_res)
            logger.info(f'Model Test Accuracy : {accuracy_score(self.y_test,self.reg_log.predict(self.testing_data))}')
            logger.info(f'Confusion Matrix : {confusion_matrix(self.y_test,self.reg_log.predict(self.testing_data))}')
            logger.info(f'Classification Report : {classification_report(self.y_test,self.reg_log.predict(self.testing_data))}')'''

            logger.info(f'Best Model Decision Tree Classifier')
            self.dt_reg = DecisionTreeClassifier(criterion='entropy')
            self.dt_reg.fit(self.training_data_res_t,self.y_train_res)
            logger.info(f'Model Test Accuracy : {accuracy_score(self.y_test, self.dt_reg.predict(self.testing_data))}')
            logger.info(f'Confusion Matrix : {confusion_matrix(self.y_test, self.dt_reg.predict(self.testing_data))}')
            logger.info(f'Classification Report : {classification_report(self.y_test, self.dt_reg.predict(self.testing_data))}')

            '''logger.info(f'RF')
            self.rf_reg = RandomForestClassifier(criterion='entropy', n_estimators=5)
            self.rf_reg.fit(self.training_data_res_t,self.y_train_res)
            logger.info(f'Model Test Accuracy : {accuracy_score(self.y_test, self.rf_reg.predict(self.testing_data))}')
            logger.info(f'Confusion Matrix : {confusion_matrix(self.y_test, self.rf_reg.predict(self.testing_data))}')
            logger.info(f'Classification Report : {classification_report(self.y_test, self.rf_reg.predict(self.testing_data))}')'''

            logger.info(f'=====Model Saving======')
            with open('Xpress_Bees.pkl','wb') as f:
                pickle.dump(self.dt_reg,f)

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

if __name__ == "__main__":
    try:
        path = 'E:\\Aspire Tech Academy Bangalore\\Data Science Tools\\Machine Learning\\Machine Learning Projects\\Xpress Bees Project\\XpressBees.xlsx'
        obj = XPRESS_BEES(path)
        obj.missing_values()
        obj.handle_outliers()
        obj.feature_selection()
        obj.cat_to_num()
        obj.merge_data()
        obj.balanced_data()
        obj.feature_scaling()
        obj.train_models()
        obj.best_model()
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')













