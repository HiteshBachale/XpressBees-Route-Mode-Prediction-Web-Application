import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import logging
from log_code import setup_logging
logger = setup_logging('train_algo')
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve

def knn_algo(X_train, y_train, X_test, y_test):
    try:
        knn_reg = KNeighborsClassifier(n_neighbors=5)
        knn_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy KNN : {accuracy_score(y_test, knn_reg.predict(X_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, knn_reg.predict(X_test))}')
        logger.info(f'classification_report : {classification_report(y_test, knn_reg.predict(X_test))}')
        global knn_pred
        knn_pred = knn_reg.predict(X_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def nb_algo(X_train, y_train, X_test, y_test):
    try:
        nb_reg = GaussianNB()
        nb_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy NB : {accuracy_score(y_test, nb_reg.predict(X_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, nb_reg.predict(X_test))}')
        logger.info(f'classification_report : {classification_report(y_test, nb_reg.predict(X_test))}')
        global nb_pred
        nb_pred = nb_reg.predict(X_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def lr_algo(X_train, y_train, X_test, y_test):
    try:
        lr_reg = LogisticRegression()
        lr_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy LR : {accuracy_score(y_test, lr_reg.predict(X_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, lr_reg.predict(X_test))}')
        logger.info(f'classification_report : {classification_report(y_test, lr_reg.predict(X_test))}')
        global lr_pred
        lr_pred = lr_reg.predict(X_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def dt_algo(X_train, y_train, X_test, y_test):
    try:
        dt_reg = DecisionTreeClassifier(criterion='entropy')
        dt_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy DT : {accuracy_score(y_test, dt_reg.predict(X_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, dt_reg.predict(X_test))}')
        logger.info(f'classification_report : {classification_report(y_test, dt_reg.predict(X_test))}')
        global dt_pred
        dt_pred = dt_reg.predict(X_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def rf_algo(X_train, y_train, X_test, y_test):
    try:
        rf_reg = RandomForestClassifier(criterion='entropy', n_estimators=5)
        rf_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy RF : {accuracy_score(y_test, rf_reg.predict(X_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, rf_reg.predict(X_test))}')
        logger.info(f'classification_report : {classification_report(y_test, rf_reg.predict(X_test))}')
        global rf_pred
        rf_pred = rf_reg.predict(X_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def best_model_using_auc_roc(X_train, y_train, X_test, y_test):
    try:
        knn_fpr, knn_tpr, knn_thre = roc_curve(y_test, knn_pred)
        nb_fpr, nb_tpr, nb_thre = roc_curve(y_test, nb_pred)
        lr_fpr, lr_tpr, lr_thre = roc_curve(y_test, lr_pred)
        dt_fpr, dt_tpr, dt_thre = roc_curve(y_test, dt_pred)
        rf_fpr, rf_tpr, rf_thre = roc_curve(y_test, rf_pred)

        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("AUC ROC Curve - All Models")
        plt.plot([0, 1], [0, 1], "k--")

        plt.plot(knn_fpr, knn_tpr, color='r', label="KNN Algo")
        plt.plot(nb_fpr, nb_tpr, color='blue', label="NB Algo")
        plt.plot(lr_fpr, lr_tpr, color='green', label="LR Algo")
        plt.plot(dt_fpr, dt_tpr, color='black', label="DT Algo")
        plt.plot(rf_fpr, rf_tpr, color='yellow', label="RF Algo")

        plt.legend(loc=0)
        plt.show()
        
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def common(X_train, y_train, X_test, y_test):
    try:
        logger.info('Giving Data to Each Function')
        logger.info('----KNN--------')
        knn_algo(X_train, y_train, X_test, y_test)
        logger.info('----NB--------')
        nb_algo(X_train, y_train, X_test, y_test)
        logger.info('----LR--------')
        lr_algo(X_train, y_train, X_test, y_test)
        logger.info('----DT--------')
        dt_algo(X_train, y_train, X_test, y_test)
        logger.info('----RF--------')
        rf_algo(X_train, y_train, X_test, y_test)
        logger.info(f'--------AUC--ROC----------')
        best_model_using_auc_roc(X_train, y_train, X_test, y_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

