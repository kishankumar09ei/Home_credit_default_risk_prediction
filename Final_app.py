# Importing required libraries for overall analysis
import streamlit as st
import warnings
import matplotlib

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline
pd.set_option('display.max_rows', None)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import gc

sns.set_style("whitegrid");
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle as pkl
import tqdm as tqdm
from random import choices
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from tqdm import tqdm
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import LabelEncoder
from scipy import stats

st.set_page_config(layout="wide")


def reduce_data_size(df):
    """
    DUe to memory constarints we will try to reduce the data size by chnaging the datatypes of columns without loosing any information
    """
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df


@st.cache(ttl=60 * 5, persist=True)
def load_data():
    df_test_preprocessed = joblib.load("FeatureEnginered/df_test_preprocessed.pkl")
    df_test = df_test = pd.read_csv(r'home-credit-default-risk/application_test.csv')

    return df_test_preprocessed, reduce_data_size(df_test)


df_test_preprocessed, df_test = load_data()


def preporcess_application_data(test):
    """
    This function take application_train|test .csv and just do some normal preprocessing
    These are just primary preprocessing
    """
    # Step 1. drop the non important columns

    # In EDA we have seen some column which were no use for predicting Target Varibale,due to low variance,and multicollineraity
    # So we will drop it

    col_to_drop = ['FLAG_MOBIL', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12',
                   'FLAG_DOCUMENT_20']
    # For building properties,we will just keep the AVG columns and remove median and mode as those are giving same infoprmation
    # For median values of bulinding properties
    avg = [col for col in test.columns if col.split("_")[-1] == 'AVG']
    median_cols = [col for col in test.columns if col.split("_")[-1] == 'MEDI']
    # for mode values of bulinding properties
    mode_cols = [col for col in test.columns if col.split("_")[-1] == 'MODE']
    non_building_mode_cols = list(
        set([i.split('_')[0] for i in mode_cols]).difference(set([i.split('_')[0] for i in avg])))

    non_building_mode_cols = [elem + "_MODE" for elem in non_building_mode_cols]

    # non_building_mode_cols

    mode_cols = [elem for elem in mode_cols if elem not in non_building_mode_cols]

    # add these to col_to_drop
    col_to_drop.extend(median_cols)
    col_to_drop.extend(mode_cols)
    # drop these columns
    test = test.drop(col_to_drop, axis=1)
    # step : 2.

    # here we will convert some column and also remove some outlier and make those NAN
    ##converting age from days to years
    test['YEARS_BIRTH'] = (-1 / 365) * test['DAYS_BIRTH']
    # we can see in test data we dont have 'XNA' category in code_gender,So we can remove such rows,also we just had 4 rows in train dat
    test = test[test['CODE_GENDER'] != 'XNA']

    # in DAYS_EMPLOYED we have some outliers as 365243,we need to remove it or better we can make it NAN

    test[test['DAYS_EMPLOYED'] == 365243]['DAYS_EMPLOYED'] = np.nan

    # similary we can do same for SOCIAL CIRCLE columns
    test[test['OBS_30_CNT_SOCIAL_CIRCLE'] > 30]['OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    test[test['OBS_60_CNT_SOCIAL_CIRCLE'] > 30]['OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan

    return test


def categorical_encoding_and_fillna_continous_application(test):
    # NAME_CONTRACT_TYPE
    NAME_CONTRACT_TYPE_encode = {'Cash loans': 1, 'Revolving loans': 0}
    test['NAME_CONTRACT_TYPE'] = test['NAME_CONTRACT_TYPE'].map(NAME_CONTRACT_TYPE_encode)

    # CODE_GENDER
    test['CODE_GENDER'] = test['CODE_GENDER'].map({'F': 1, 'M': 2})

    # FLAG_OWN_CAR
    test['FLAG_OWN_CAR'] = test['FLAG_OWN_CAR'].map({'N': 1, 'Y': 0})

    # FLAG_OWN_REALTY
    test['FLAG_OWN_REALTY'] = test['FLAG_OWN_REALTY'].map({'N': 0, 'Y': 1})

    # NAME_TYPE_SUITE
    test['NAME_TYPE_SUITE'].fillna('Unaccompanied', inplace=True)

    le_NAME_TYPE_SUITE = joblib.load("required_files/le_NAME_TYPE_SUITE.joblib")
    test['NAME_TYPE_SUITE'] = le_NAME_TYPE_SUITE.transform(test['NAME_TYPE_SUITE'])

    # NAME_INCOME_TYPE
    le_NAME_INCOME_TYPE = joblib.load("required_files/le_NAME_INCOME_TYPE.joblib")

    test['NAME_INCOME_TYPE'] = le_NAME_INCOME_TYPE.transform(test['NAME_INCOME_TYPE'])
    # NAME_EDUCATION_TYPE
    map_NAME_EDUCATION_TYPE = {'Secondary / secondary special': 4,
                               'Higher education': 2,
                               'Incomplete higher': 3,
                               'Lower secondary': 5,
                               'Academic degree': 1}

    test['NAME_EDUCATION_TYPE'] = test['NAME_EDUCATION_TYPE'].map(map_NAME_EDUCATION_TYPE)

    # NAME_FAMILY_STATUS
    map_NAME_FAMILY_STATUS = {'Married': 5,
                              'Single / not married': 2,
                              'Civil marriage': 3,
                              'Separated': 2,
                              'Widow': 1,
                              'Unknown': 0}
    test['NAME_FAMILY_STATUS'] = test['NAME_FAMILY_STATUS'].map(map_NAME_FAMILY_STATUS)

    # NAME_HOUSING_TYPE
    map_NAME_HOUSING_TYPE = {'House / apartment': 5,
                             'With parents': 4,
                             'Municipal apartment': 2,
                             'Rented apartment': 3, 'Office apartment': 1,
                             'Co-op apartment': 1}

    test['NAME_HOUSING_TYPE'] = test['NAME_HOUSING_TYPE'].map(map_NAME_HOUSING_TYPE)

    # OCCUPATION_TYPE
    map_OCCUPATION_TYPE = {'IT staff': 0,
                           'HR staff': 0,
                           'Realty agents': 0,
                           'Secretaries': 0,
                           'Waiters/barmen staff': 0,
                           'Private service staff': 0,
                           'Low-skill Laborers': 1,
                           'Cleaning staff': 1,
                           'Accountants': 1,
                           'Medicine staff': 2,
                           'Cooking staff': 2,
                           'High skill tech staff': 2,
                           'Security staff': 2,
                           'Managers': 3,
                           'Core staff': 3,
                           'Drivers': 3,
                           'Sales staff': 4,
                           'Laborers': 5}

    test['OCCUPATION_TYPE'] = test['OCCUPATION_TYPE'].map(map_OCCUPATION_TYPE)

    # filling na values 1 as these occupation with no values can belong to category 1
    test['OCCUPATION_TYPE'] = test['OCCUPATION_TYPE'].fillna(1)

    # WEEKDAY_APPR_PROCESS_START: we will just drop this column because its no use
    test.drop('WEEKDAY_APPR_PROCESS_START', axis=1, inplace=True)

    # ORGANIZATION_TYPE,we can catregorize in it in 3
    # ORGANIZATION_TYPE
    map_org_type_encode = joblib.load("required_files/map_org_type_encode.joblib")

    test['ORGANIZATION_TYPE'] = test['ORGANIZATION_TYPE'].map(map_org_type_encode)

    #     #FONDKAPREMONT_MODE
    test['FONDKAPREMONT_MODE'].fillna('Not specified', inplace=True)

    # we can label encode it
    le_FONDKAPREMONT_MODE = joblib.load("required_files/le_FONDKAPREMONT_MODE.joblib")

    test['FONDKAPREMONT_MODE'] = le_FONDKAPREMONT_MODE.transform(test['FONDKAPREMONT_MODE'])

    # HOUSETYPE_MODE
    map_HOUSETYPE_MODE = {'Not specified': 2, 'block of flats': 1, 'specific housing': 0, 'terraced house': 0}
    test['HOUSETYPE_MODE'] = test['HOUSETYPE_MODE'].map(map_HOUSETYPE_MODE)

    # filling na as 'Not specified':2
    test['HOUSETYPE_MODE'] = test['HOUSETYPE_MODE'].fillna(2)

    # WALLSMATERIAL_MODE
    test['WALLSMATERIAL_MODE'] = test['WALLSMATERIAL_MODE'].fillna("Not Specified")

    map_wallsmaterial_type = {'Monolithic': 0,
                              'Others': 0,
                              'Mixed': 0,
                              'Wooden': 1,
                              'Block': 1,
                              'Panel': 1,
                              'Stone, brick': 1,
                              'Not Specified': 2}
    test['WALLSMATERIAL_MODE'] = test['WALLSMATERIAL_MODE'].map(map_wallsmaterial_type)

    # EMERGENCYSTATE_MODE

    test['EMERGENCYSTATE_MODE'] = test['EMERGENCYSTATE_MODE'].fillna("No")
    test['EMERGENCYSTATE_MODE'] = test['EMERGENCYSTATE_MODE'].map({'Yes': 1, 'No': 0})
    # Filling Na Values for continous columns
    # AMT_GOODS_PRICE : fill it with same value of AMT_credit
    test['AMT_GOODS_PRICE'] = test['AMT_GOODS_PRICE'].fillna(test['AMT_CREDIT'])

    # OWN_CAR_AGE,fillna with zero because they dont have car
    test['OWN_CAR_AGE'] = test['OWN_CAR_AGE'].fillna(0)

    # DAYS_LAST_PHONE_CHANGE
    test['DAYS_LAST_PHONE_CHANGE'] = test['DAYS_LAST_PHONE_CHANGE'].fillna(0)
    # CNT_FAM_MEMBERS
    test['CNT_FAM_MEMBERS'] = test['CNT_FAM_MEMBERS'].fillna(0)
    # AMT_ANNUITY;we suppose that where AMT_ANNUITY is equal to  AMT_CREDIT without considering interest and repayment
    # period as 1 year
    test['AMT_ANNUITY'] = test['AMT_ANNUITY'].fillna(test['AMT_CREDIT'])

    # fill na for 'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3' with 0 i.e zero income from thate externel income source
    test['EXT_SOURCE_1'] = test['EXT_SOURCE_1'].fillna(0)
    test['EXT_SOURCE_2'] = test['EXT_SOURCE_2'].fillna(0)
    test['EXT_SOURCE_3'] = test['EXT_SOURCE_3'].fillna(0)
    # TOTALAREA_MODE: filling it with  highest correlation colum LIVINGAREA_AVG
    # train['TOTALAREA_MODE']= np.where(train['TOTALAREA_MODE'].isnull(), train['LIVINGAREA_AVG'], train['TOTALAREA_MODE'])

    # For Avg columns
    building_properties_avg = [col for col in test.columns if col.split("_")[-1] == 'AVG']
    for i in building_properties_avg:
        mean = test[i].median()
        # train[i]= train[i].fillna(mean)
        test[i] = test[i].fillna(mean)
    # AMT_REQ_CREDIT_BUREAU_HOUR    13.501806
    test['AMT_REQ_CREDIT_BUREAU_HOUR'] = test['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0)
    # AMT_REQ_CREDIT_BUREAU_DAY     13.501806
    test['AMT_REQ_CREDIT_BUREAU_DAY'] = test['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(0)
    # AMT_REQ_CREDIT_BUREAU_WEEK    13.501806
    test['AMT_REQ_CREDIT_BUREAU_WEEK'] = test['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(0)
    # AMT_REQ_CREDIT_BUREAU_MON     13.501806
    test['AMT_REQ_CREDIT_BUREAU_MON'] = test['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0)
    # AMT_REQ_CREDIT_BUREAU_QRT     13.501806
    test['AMT_REQ_CREDIT_BUREAU_QRT'] = test['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0)
    # AMT_REQ_CREDIT_BUREAU_YEAR    13.501806
    test['AMT_REQ_CREDIT_BUREAU_YEAR'] = test['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0)
    # OBS_30_CNT_SOCIAL_CIRCLE       0.332025
    test['OBS_30_CNT_SOCIAL_CIRCLE'] = test['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(0)
    # DEF_30_CNT_SOCIAL_CIRCLE       0.332025
    test['DEF_30_CNT_SOCIAL_CIRCLE'] = test['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(0)
    # OBS_60_CNT_SOCIAL_CIRCLE       0.332025
    test['OBS_60_CNT_SOCIAL_CIRCLE'] = test['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(0)
    # DEF_60_CNT_SOCIAL_CIRCLE       0.332025
    test['DEF_60_CNT_SOCIAL_CIRCLE'] = test['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(0)
    test['TOTALAREA_MODE'] = np.where(test['TOTALAREA_MODE'].isna(), test['LIVINGAREA_AVG'], test['TOTALAREA_MODE'])

    return test


def application_data_feature_engineering(df):
    """
    Here we will create some intersting features,These are basically domain knowledge and some random features
    """
    # 1. for client EMployment and client Birth#####################################################
    df['PER_DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['DAYS_UNEMPLOYED'] = abs(df['DAYS_BIRTH']) - abs(df['DAYS_EMPLOYED'])
    df['PER_DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['DAYS_UNEMPLOYED'] = abs(df['DAYS_BIRTH']) - abs(df['DAYS_EMPLOYED'])
    df["OWN_CAR_AGE_RATIO"] = df["OWN_CAR_AGE"] / df["DAYS_BIRTH"]
    df["DAYS_ID_PUBLISHED_RATIO"] = df["DAYS_ID_PUBLISH"] / df["DAYS_BIRTH"]
    df["DAYS_REGISTRATION_RATIO"] = df["DAYS_REGISTRATION"] / df["DAYS_BIRTH"]
    df["DAYS_LAST_PHONE_CHANGE_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]
    # 2. clients INCOME#################################################################
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 0.001)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 0.001)
    df['GOODS_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_INCOME_TOTAL'] + 0.001)
    df['INCOME_EXT1_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['EXT_SOURCE_1'] + 0.001)
    df['INCOME_EXT2_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['EXT_SOURCE_2'] + 0.001)
    df['INCOME_EXT3_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['EXT_SOURCE_3'] + 0.001)
    df['INCOME_ANNUITY_DIFF'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']
    # percentage income of person and the credit amount
    df['INCOME_PER_CAPITA'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    ########For AMT_Credit and AMT_ANNUITY####################################################
    # percentage income of person and the credit amount
    df['INCOME_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    # Amount paid for previous loan appication every month decided by the number of day employed
    df['ANNUITY_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['AMT_ANNUITY']

    df['AMT_CREDIT_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['AMT_CREDIT']
    # Anually paid amount to amount credited
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    df['PAYMENT_RATE_INV'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    df['PAY_TOWARDS_LOAN'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']
    df['CREDIT_EXT1_RATIO'] = df['AMT_CREDIT'] / (df['EXT_SOURCE_1'] + 0.001)
    df['CREDIT_EXT2_RATIO'] = df['AMT_CREDIT'] / (df['EXT_SOURCE_2'] + 0.001)
    df['CREDIT_EXT3_RATIO'] = df['AMT_CREDIT'] / (df['EXT_SOURCE_2'] + 0.001)

    # FOR OWN_CAR_AGE##########################################################################
    df['CAR_EMPLOYED_DIFF'] = df['OWN_CAR_AGE'] - df['DAYS_EMPLOYED']
    df['CAR_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_EMPLOYED'] + 0.00001)
    df['CAR_AGE_DIFF'] = abs(df['DAYS_BIRTH']) - abs(df['OWN_CAR_AGE'])
    df['CAR_AGE_RATIO'] = df['OWN_CAR_AGE'] / (abs(df['DAYS_BIRTH']))

    # FOR Family members #############################################################################
    df["CNT_ADULTS"] = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]
    df['CHILDREN_RATIO'] = (df['CNT_CHILDREN'] + 0.0001) / (df['CNT_FAM_MEMBERS'])
    df['LOG_CNT_CHILDREN'] = np.log(df['CNT_CHILDREN'])
    df['LOG_CNT_FAM_MEMBERS'] = np.log(df['CNT_FAM_MEMBERS'])
    # FOR FLAG DOCMENTS
    flag_doc_cols = [col for col in df.columns if "FLAG_DOCUMENT_" in col]
    df['CNT_FLAG_DOCS'] = df[flag_doc_cols].sum(axis=1)
    # for "NOT_LIVE" and "NOT WORK",These are the column where Flag=1 if there is a adress missmatch
    address_missmatch_cols = [col for col in df.columns if ("NOT_LIVE" in col) or ("NOT_WORK" in col)]
    df['ADDRESS_MISMATCH'] = df[flag_doc_cols].sum(axis=1)
    # Even i know the AMT_REQ_CREDIT_BUREAU are no use here but will just take mean of these columns
    AMT_Req_cb_cols = [col for col in df.columns if "AMT_REQ_CREDIT_BUREAU" in col]
    df['AMT_REQ_CREDIT_BUREAU_MEAN'] = df[flag_doc_cols].mean(axis=1)
    df['AMT_REQ_CREDIT_BUREAU_SUM'] = df[flag_doc_cols].sum(axis=1)
    df['AMT_ENQ_CREDIT_RATIO'] = df['AMT_REQ_CREDIT_BUREAU_SUM'] / (df['AMT_CREDIT'] + 0.00001)

    # for Phone/Email contant
    df['All_CONTACTS'] = (
        (df[['FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']]).sum(axis=1))
    # for days ID ,Registration changed
    df['MAX_DAYS_CHANGED'] = ((df[['DAYS_ID_PUBLISH', 'DAYS_REGISTRATION']]).max(axis=1))
    # EXT_SOURCE_COLUMNS
    df['EXT_SOURCE_SUM'] = (df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]).sum(axis=1)
    df['EXT_SOURCE_MEAN'] = (df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]).mean(axis=1)

    df['EXT_SOURCE_MEDIAN'] = (df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]).median(axis=1)

    df['EXT_SOURCE_MIN'] = (df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]).min(axis=1)

    df['EXT_SOURCE_MAX'] = (df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]).max(axis=1)
    # BUILDING PROPERT SCORE scores,we have only considered the AVG ones because mean
    avg = [col for col in df.columns if col.split("_")[-1] == 'AVG']
    df['BUILDING_PROPERTIES_AVG_SUM'] = df[avg].sum(axis=1)
    # client's social surroundings OBSERVED And DEFAULTED
    social_surr_cols = [col for col in df.columns if "CNT_SOCIAL_CIRCLE" in col]
    df['CNT_SOCIAL_CIRCLE_MEAN'] = df[social_surr_cols].sum(axis=1)

    # Now we will create some features based on contionous columns and aggregated by categorical columns
    # We wont create any column which have some reference to "TARGET" VARIABLE,because TEST data wont have it
    # We have the categorcial column and some continous column which are actually categorical column ,We will include both
    # we have seen some interaction between differenet categorical variable ,SO we will try to group on some categorical varaible
    interaction_cols_for_aggregation_on = [['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'],
                                           ['CODE_GENDER', 'NAME_CONTRACT_TYPE'],
                                           ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY'],
                                           ['CODE_GENDER', 'FLAG_OWN_REALTY'],
                                           ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'CODE_GENDER'],
                                           ['NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE'],
                                           ['FONDKAPREMONT_MODE', 'NAME_INCOME_TYPE'],
                                           ['HOUSETYPE_MODE', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE'],
                                           ['EMERGENCYSTATE_MODE', 'NAME_INCOME_TYPE'], ['REGION_RATING_CLIENT']]

    # Contionious column for aggregation,we are not using all we are just using some important one,
    contiouns_cols_for_agrregation_of = {'AMT_ANNUITY': ['mean', 'max', 'min'],
                                         'ANNUITY_INCOME_RATIO': ['mean', 'max', 'min'],
                                         'DAYS_UNEMPLOYED': ['mean', 'min'],
                                         'AMT_INCOME_TOTAL': ['mean', 'max', 'min'],
                                         'BUILDING_PROPERTIES_AVG_SUM': ['mean', 'max', 'min'],
                                         'EXT_SOURCE_MEAN': ['mean', 'max', 'min'],
                                         'EXT_SOURCE_1': ['mean', 'max', 'min'],
                                         'EXT_SOURCE_2': ['mean', 'max', 'min'],
                                         'EXT_SOURCE_3': ['mean', 'max', 'min']}

    for group_col in interaction_cols_for_aggregation_on:
        # grouping on categorical feature interaction
        grouped_data = df.groupby(group_col).agg(contiouns_cols_for_agrregation_of)
        grouped_data.columns = ['_'.join(i).upper() + '_AGG_BY_' + '_'.join(group_col) for i in grouped_data.columns]
        # merging with data
        df = df.join(grouped_data, on=group_col)
    # Some Log and box cox transform
    log_vars = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "AMT_ANNUITY"]
    for i in log_vars:
        df['LOG_' + str(i)] = np.log(abs(df[i]) + 1)

    box_cox_vars = ['DAYS_EMPLOYED', 'YEARS_BIRTH', 'DAYS_REGISTRATION', 'OWN_CAR_AGE', 'EXT_SOURCE_1', 'EXT_SOURCE_1',
                    'EXT_SOURCE_1']

    for i in box_cox_vars:
        df['BOXCOX_' + str(i)], _ = stats.boxcox(abs(df[i]) + 1)

    # df= pd.get_dummies(df)
    df.columns = ["app_" + col.upper() if col not in ["TARGET", "SK_ID_CURR"] else col for col in df.columns]

    return df


def merge_with_other_tables(df_test):
    bureau_merged_agg = joblib.load("FeatureEnginered/bureau_merged_agg.pkl")
    df_test = df_test.merge(bureau_merged_agg, how='left', on='SK_ID_CURR')
    del bureau_merged_agg
    gc.collect()
    df_prev_app_data = joblib.load('FeatureEnginered/df_prev_app_data.pkl')
    df_test = df_test.merge(df_prev_app_data, how='left', on='SK_ID_CURR')
    del df_prev_app_data
    gc.collect()
    POS_CASH_agg = joblib.load('FeatureEnginered/POS_CASH_agg.pkl')
    df_test = df_test.merge(POS_CASH_agg, how='left', on='SK_ID_CURR')
    del POS_CASH_agg
    gc.collect()
    cc_bal_agg = joblib.load("FeatureEnginered/cc_bal_agg.pkl")
    df_test = df_test.merge(cc_bal_agg, how='left', on='SK_ID_CURR')
    del cc_bal_agg
    gc.collect()
    install_pay_agg = joblib.load("FeatureEnginered/install_pay_agg.pkl")
    df_test = df_test.merge(install_pay_agg, how='left', on='SK_ID_CURR')
    del install_pay_agg
    gc.collect()
    return reduce_data_size(df_test)


def imputeNa_preprocessing(df_test):
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    more_than_75_per_NA_cols = joblib.load('required_files/more_than_75_per_NA_cols.pkl')
    df_test = df_test.drop(columns=more_than_75_per_NA_cols)
    morethan30_lessthan75_per_NA_cols = joblib.load('required_files/morethan30_lessthan75_per_NA_cols.pkl')
    lessthan30_per_NA_cols = joblib.load('required_files/lessthan30_per_NA_cols.pkl')
    imputer_mean = joblib.load('required_files/imputer_mean.pkl')
    mean_imputed_df_test = imputer_mean.transform(df_test[lessthan30_per_NA_cols])
    df_test.loc[:, lessthan30_per_NA_cols] = mean_imputed_df_test.copy()
    del mean_imputed_df_test
    col_lgbr_predict = joblib.load('required_files/col_lgbr_predict.pkl')

    for col in morethan30_lessthan75_per_NA_cols:
        s1 = {'SK_ID_CURR'}
        s2 = set(morethan30_lessthan75_per_NA_cols)
        s3 = s2.union(s1)

        test_cols = list(set(df_test.columns) - s3) + [col]
        test = df_test[df_test[col].isnull()][test_cols]
        X_test = test.drop(col, axis=1).values
        lgbr = col_lgbr_predict[col]
        df_test.loc[df_test[col].isnull(), col] = lgbr.predict(X_test)

    del test_cols, imputer_mean, col_lgbr_predict
    gc.collect()

    return df_test


def final_feature_engineering(df_test):
    # previous applications columns
    prev_app_cols = [col for col in df_test.columns if 'PREVAPP_' in col]

    # For Amt_Annuity
    prev_app_annuity_cols = [col for col in prev_app_cols if "AMT_ANNUITY" in col]

    for col in prev_app_annuity_cols:
        df_test["FINAL_" + str(col) + "_RATIO"] = df_test[col] / (df_test['app_AMT_ANNUITY'] + 0.0001)
        df_test["FINAL_" + str(col) + "_INCOME_RATIO"] = df_test[col] / (df_test['app_AMT_INCOME_TOTAL'] + 0.0001)

    # For AMT_CREDIT

    prev_app_CREDIT_cols = [col for col in prev_app_cols if "AMT_CREDIT" in col]
    for col in prev_app_CREDIT_cols:
        df_test["FINAL_" + str(col) + "_RATIO"] = df_test[col] / (df_test['app_AMT_CREDIT'] + 0.0001)
        df_test["FINAL_" + str(col) + "_INCOME_RATIO"] = df_test[col] / (df_test['app_AMT_INCOME_TOTAL'] + 0.0001)

    # For AMT_GOODS
    prev_app_GOODS_cols = [col for col in prev_app_cols if "AMT_GOODS" in col]
    # There was no income column in prev_app data
    for col in prev_app_GOODS_cols:
        # df_test[str(col)+"_RATIO"]= df_test[col]/(df_test['app_AMT_GOODS_PRICE']+0.0001)
        df_test["FINAL_" + str(col) + "_INCOME_RATIO"] = df_test[col] / (df_test['app_AMT_INCOME_TOTAL'] + 0.0001)

    # del prev_app_annuity_cols,prev_app_GOODS_cols,prev_app_CREDIT_cols

    # bureau and bureau_balance columns

    # we need to handle this data different ,we need to go back to EDA and see how this worked with respect to app data
    bbl_cols = [col for col in df_test.columns if 'BBL_' in col]
    # DAYS CREDIT
    bbl_DAYS_CREDIT_cols = [col for col in bbl_cols if
                            'BBL_DAYS_CREDIT' in col and 'ENDDATE' not in col and 'UPDATE' not in col]
    for col in bbl_DAYS_CREDIT_cols:
        df_test["FINAL_" + str(col) + "_EMPLOYMENT_DIFF"] = df_test[col] - df_test['app_DAYS_EMPLOYED']
        df_test["FINAL_" + str(col) + "_REGISTRATION_DIFF"] = df_test[col] - df_test['app_DAYS_REGISTRATION']

    # AMT CREDIT_Overdue
    bbl_AMT_CREDIT_OD_cols = [col for col in bbl_cols if 'AMT_CREDIT' in col and 'OVERDUE' in col]

    for col in bbl_AMT_CREDIT_OD_cols:
        df_test["FINAL_" + str(col) + "_INCOME_RATIO"] = df_test[col] - df_test['app_AMT_INCOME_TOTAL']

        # Some other Features from Kaggle disscussions
    df_test["FINAL_BBL_AMT_ANUUITY_RATIO"] = (df_test['app_AMT_ANNUITY'] + 0.00001) / (
            df_test['BBL_AMT_ANNUITY_MEAN'] + 0.00001)
    df_test["FINAL_BBL_AMT_CREDIT_RATIO"] = (df_test['app_AMT_CREDIT'] + 0.00001) / (
            df_test['BBL_AMT_CREDIT_SUM_MEAN'] + 0.00001)

    del bbl_cols, bbl_DAYS_CREDIT_cols, bbl_AMT_CREDIT_OD_cols

    gc.collect()

    # credit_card_balance columns

    CC_bal_cols = [col for col in df_test.columns if 'CCBAL_' in col]

    # AMT_BALANCE

    CC_bal_AMT_BALANCE_cols = [col for col in CC_bal_cols if 'MONTHS_BALANCE' in col]

    for col in CC_bal_AMT_BALANCE_cols:
        df_test["FINAL_" + str(col) + "_ANNUITY_RATIO"] = df_test[col] / (df_test['app_AMT_ANNUITY'] + 0.0001)

    # AMT_RECIEVABLE ,we are including here ALL recievable amounts like Prinicipal,Amount and Total Recievable
    # ['CCBAL_AMT_RECEIVABLE_PRINCIPAL_MEAN',
    #  'CCBAL_AMT_TOTAL_RECEIVABLE_MEAN',
    #  'CCBAL_AMT_RECEIVABLE_SUM_MEAN']
    CC_bal_RECEIVABLE_cols = [col for col in CC_bal_cols if 'RECEIVABLE' in col]

    for col in CC_bal_RECEIVABLE_cols:
        df_test["FINAL_" + str(col) + "_ANNUITY_RATIO"] = df_test[col] / (df_test['app_AMT_ANNUITY'] + 0.0001)
        df_test["FINAL_" + str(col) + "_INCOME_RATIO"] = df_test[col] / (df_test['app_AMT_INCOME_TOTAL'] + 0.0001)

    del CC_bal_cols, CC_bal_AMT_BALANCE_cols, CC_bal_RECEIVABLE_cols

    # installments_payments columns
    INSTLPAY_cols = [col for col in df_test.columns if 'INSTLPAY_' in col]

    # AMT INSTALLMENT
    INSTLPAY_AMT_INSTALMENT_cols = [col for col in INSTLPAY_cols if 'AMT_INSTALMENT' in col]

    for col in INSTLPAY_AMT_INSTALMENT_cols:
        df_test["FINAL_" + str(col) + "_INCOME_RATIO"] = df_test[col] / (df_test['app_AMT_INCOME_TOTAL'] + 0.0001)

    # AMT_PAYMENT

    INSTLPAY_AMT_PAYMENT_cols = [col for col in INSTLPAY_cols if 'AMT_PAYMENT' in col]

    for col in INSTLPAY_AMT_PAYMENT_cols:
        # https://www.kaggle.com/c/home-credit-default-risk/discussion/64821

        df_test["FINAL_" + str(col) + "_ANNUITY_RATIO"] = df_test['app_AMT_ANNUITY'] / (df_test[col] + 0.0001)
        df_test["FINAL_" + str(col) + "_INCOME_RATIO"] = df_test[col] / (df_test['app_AMT_INCOME_TOTAL'] + 0.0001)

    del INSTLPAY_cols, INSTLPAY_AMT_INSTALMENT_cols, INSTLPAY_AMT_PAYMENT_cols

    # we can create a dummy column for CIBIL SCore
    cibil_cols = ['INSTLPAY_PAYMENT_INSTALLEMENT_NUM_DIFF_MEAN', 'INSTLPAY_DAYS_LATE_PAYMENT_MEAN',
                  'INSTLPAY_FLAG_LATE_PAYMENT_SUM', 'INSTLPAY_FLAG_LESS_PAYMENT_SUM', 'INSTLPAY_FLAG_NO_PAYMENT_SUM',
                  'BBL_YEAR_CREDIT_MEAN', 'BBL_FLAG_SECURED_LOAN_SUM', 'BBL_FLAG_UNSECURED_LOAN_SUM',
                  'app_AMT_REQ_CREDIT_BUREAU_WEEK']
    df_cibil = df_test[cibil_cols]

    scaler_cibil = MinMaxScaler(feature_range=(0, 1))

    scaler_cibil.fit(df_cibil)

    df_cibil = scaler_cibil.transform(df_cibil)

    df_cibil = pd.DataFrame(data=df_cibil, columns=cibil_cols)

    df_cibil['cibil_score'] = ((0.05 * df_cibil[cibil_cols[0]] + 0.05 * df_cibil[cibil_cols[1]] + 0.1 * df_cibil[
        cibil_cols[2]] + 0.1 * df_cibil[cibil_cols[3]] + 0.05 * df_cibil[cibil_cols[4]])
                               + (0.25 * df_cibil[cibil_cols[5]] + 0.10 * df_cibil[cibil_cols[6]] + 0.15 * df_cibil[
                cibil_cols[7]] +
                                  0.2 * df_cibil[cibil_cols[8]])) * 100

    df_test['cibil_score'] = df_cibil['cibil_score'].copy()
    # df_test.replace([np.inf, -np.inf], 0,inplace=True)

    del df_cibil
    gc.collect()

    return reduce_data_size(df_test)


def final_feature_selection(df_test):
    single_value_columns = joblib.load("FeatureEnginered/single_value_columns.pkl")
    single_val_cols = [i for i in single_value_columns if i in df_test.columns]
    # droopping those columns

    df_test = df_test.drop(single_val_cols, axis=1)

    # recursive feature selection columns
    important_features = joblib.load("FeatureEnginered/important_features.pkl")
    important_features = ['SK_ID_CURR'] + important_features
    imp_feat = [i for i in important_features if i in df_test.columns]
    left_over_features = set(important_features) - set(imp_feat)
    df_test = df_test[imp_feat]
    for i in left_over_features:
        df_test[str(i)] = 0
    # important_features=
    df_test = df_test[important_features]

    return reduce_data_size(df_test)


def data_prepare_final(df):
    df = preporcess_application_data(df)
    df = categorical_encoding_and_fillna_continous_application(df)
    df = application_data_feature_engineering(df)
    df = merge_with_other_tables(df)
    df = imputeNa_preprocessing(df)
    df = final_feature_engineering(df)
    df = final_feature_selection(df)

    return reduce_data_size(df)


def predict_final(random_Sample):
    id_ = random_Sample['SK_ID_CURR'].values[0]
    st.write("The Customer id is : \n ", str(id_))
    st.write("Let see application data for this client")
    st.write(df_test[df_test["SK_ID_CURR"] == id_])
    data = random_Sample.drop('SK_ID_CURR', axis=1)
    lgbm_clf = joblib.load('saved_models/lgbm_clf.pkl')
    prediction_proba = lgbm_clf.predict_proba(data)[0]
    if lgbm_clf.predict(data)[0] == 0:
        prediction = "Non-defaulter"
    else:
        prediction = "Defaulter"
    st.write("Non defaulter probability is :", round(prediction_proba[0], 3), " and defaulter probability is :",
             round(prediction_proba[1], 3))
    st.write("\n So our model predicts customer id ", str(id_), "could be ", prediction)
    #feat_importances = pd.Series(lgbm_clf.feature_importances_, index=data.columns)
    #most_imp_features = feat_importances.nlargest(10).sort_values(ascending=False).index.to_list()
    #st.write("Most important Features values are : \n")
    important_feature_dict = dict()
    important_feature_dict['app_PAYMENT_RATE_INV'] = "PAYMENT_RATE: Ratio of Amount credit and Loan Annuity"
    important_feature_dict['app_EXT_SOURCE_2'] = "EXT_SOURCE_2: Second Extra Income Source"
    important_feature_dict['app_EXT_SOURCE_3'] = "EXT_SOURCE_3: Third Extra Income Source"
    important_feature_dict['app_EXT_SOURCE_MAX'] = "EXT_SOURCE_MAX:Maximum of all Extra income sources"
    important_feature_dict['app_REGION_POPULATION_RELATIVE'] = "REGION_POPULATION_RELATIVE: Population of Region where client is living"
    important_feature_dict['INSTLPAY_INSTALLMENT_PAYMENT_DIFF_MEAN'] = "INSTALLMENT_PAYMENT_DIFF: Diff. between Amount Installment and Amount Payment"
    st.write("Most important Features values are : \n")
    for i, j in important_feature_dict.items():
        st.write("\t \t", j, ":", data[i].values[0], "\n")


#st.image("ai-logo-new.png")
st.image("ai-logo-new.png")
st.title("Home-Credit Loan defaulter prediction")
st.subheader("Business Problem")
st.markdown(
    '* Home Credit B.V. is an international non-bank financial institution founded in 1997 in the Czech Republic and headquartered in Netherlands. The company operates in 9 countries and focuses on installment lending primarily to people with little or no credit history.')
st.markdown('* There is a risk associated with each offered loan product.')
st.markdown(
    '* Our main goal is , given the data of a client ,we need to predict if he/she could be potential defaulter or not.')

st.write("Sample client's data to predict : \n")


@st.cache(ttl=60 * 5, suppress_st_warning=True)
def return_sample_Data():
    sample_data = pd.read_csv(r'home-credit-default-risk/application_test.csv', nrows=10)
    return sample_data


st.dataframe(return_sample_Data())
# user_input = st.text_input("Enter the SK_ID_CURR",100001)
try:
    Sample_index = int(st.text_input("Enter a no. between 0 to 48743 as index of data for the client \n", 100))
    st.write("\n")
    # getting the sample data for the choosen index
    Sample = df_test_preprocessed.iloc[Sample_index:Sample_index + 1, :]
    # st.write("Lets look at the preprocessed data for this client")
    predict_final(Sample)
except:
    st.write("Please enter a number between 0 to 48743 ")
