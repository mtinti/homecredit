# Forked from excellent kernel : https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
# From Kaggler : https://www.kaggle.com/jsaguiar
# Just added a few features so I thought I had to make release it as well...

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import xgbfir
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm


# PREPROCESSING BLOCK -------------------------------------------------------------------------------
def reduce_mem_usage(df, skip_cols_pattern='SK_ID_'):
    """ 
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in tqdm(df.columns):

        if skip_cols_pattern in col:
            print(f"don't optimize index {col}")

        else:
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
            else:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


warnings.simplefilter(action='ignore')
#PATH = '../input/'
#EXT = ''

def trendline(data):
    #order=1
    coeffs = np.polyfit(data.values, np.arange(0, data.shape[0], 1), 1)
    slope = coeffs[-2]
    return float(slope)

PATH = 'input/'
EXT = '.zip'
NUM_FOLDS = 12
STRATIFIED = False
OUT = '2level_random/'

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

'''
agg_cols = ['region', 'city', 'parent_category_name', 'category_name',
            'image_top_1', 'user_type','item_seq_number','day_of_month','day_of_week'];
for c in tqdm(agg_cols):
    gp = tr.groupby(c)['deal_probability']
    mean = gp.mean()
    std  = gp.std()
    data[c + '_deal_probability_avg'] = data[c].map(mean)
    data[c + '_deal_probability_std'] = data[c].map(std)

for c in tqdm(agg_cols):
    gp = tr.groupby(c)['price']
    mean = gp.mean()
    data[c + '_price_avg'] = data[c].map(mean)
'''

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv(PATH+'application_train.csv'+EXT, nrows= num_rows)
    test_df = pd.read_csv(PATH+'application_test.csv'+EXT, nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    df['EXT_SOURCE_1p']=pd.qcut(df['EXT_SOURCE_1'],100,labels=False,duplicates='drop')  
    df['EXT_SOURCE_2p']=pd.qcut(df['EXT_SOURCE_2'],100,labels=False,duplicates='drop')
    df['EXT_SOURCE_3p']=pd.qcut(df['EXT_SOURCE_3'],100,labels=False,duplicates='drop')
    df['EXT_SOURCE_3pm']=df[['EXT_SOURCE_1p','EXT_SOURCE_2p','EXT_SOURCE_3p']].mean(axis=1)
    #df['na_ext'] =  df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].isnull().sum(axis=1)
    df['EXT_SOURCE_1'] = df['EXT_SOURCE_1'].fillna(df['EXT_SOURCE_1'].median())
    df['EXT_SOURCE_2'] = df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].median())
    df['EXT_SOURCE_3'] = df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].median())

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    #df['fgg']=df['EXT_SOURCE_2']*df['EXT_SOURCE_3']
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH']+abs(df['DAYS_BIRTH'].min()))
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_BIRTH']+abs(df['DAYS_BIRTH'].min()))
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_EMPLOYED']+abs(df['DAYS_EMPLOYED'].min()))
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH']+abs(df['DAYS_BIRTH'].min()))
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_EMPLOYED']+abs(df['DAYS_EMPLOYED'].min()))
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['NEW_CREDIT_TO_ANNUITY_RATIO_GOODS_PRICE'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY']+df['AMT_GOODS_PRICE'])
    df['AMT_CREDIT'] =  np.log10(df['AMT_CREDIT'] )  
    df['AMT_ANNUITY'] =  np.log10(df['AMT_ANNUITY'] ) 
    df['AMT_GOODS_PRICE']  =  np.log10(df['AMT_GOODS_PRICE'] )
    #df['tt']  =  df['DAYS_EMPLOYED'] * df['EXT_SOURCE_1']
    #del df['EXT_SOURCE_1'],df['EXT_SOURCE_2'],df['EXT_SOURCE_3']
    # Categorical features with Binary encode (0 or 1; two categories)
    ##DAYS_EMPLOYED|NAME_EDUCATION_TYPE_Secondary / secondary special
    #DAYS_EMPLOYED|EXT_SOURCE_1  
    '''
    for c in tqdm(['NAME_EDUCATION_TYPE']):
        gp = df.groupby(c)['DAYS_EMPLOYED']
        mean = gp.mean()
        std  = gp.std()
        df[c + '_NAME_EDUCATION_TYPE_EMPLOYED'] = df[c].map(mean)
        #df[c + '_AMT_ANNUITY_f1_std'] = df[c].map(std)    
    '''
    for c in tqdm(['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_EDUCATION_TYPE']):
        gp = df.groupby(c)['AMT_ANNUITY']
        mean = gp.mean()
        std  = gp.std()
        df[c + '_AMT_ANNUITY_f1_mean'] = df[c].map(mean)
        df[c + '_AMT_ANNUITY_f1_std'] = df[c].map(std)    
    
    for c in tqdm(['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']):
        gp = df.groupby(c)['AMT_CREDIT']
        mean = gp.mean()
        std  = gp.std()
        df[c + '_AMT_CREDIT_f1_mean'] = df[c].map(mean)
        df[c + '_AMT_CREDIT_f1_std'] = df[c].map(std)      

    for c in tqdm(['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']):
        gp = df.groupby(c)['AMT_GOODS_PRICE']
        mean = gp.mean()
        std  = gp.std()
        df[c + '_AMT_GOODS_PRICE_f1_mean'] = df[c].map(mean)
        df[c + '_AMT_GOODS_PRICE_f1_std'] = df[c].map(std) 
    
    df['AMT_CREDIT'] =  np.log10(df['AMT_CREDIT'] )  
    df['AMT_ANNUITY'] =  np.log10(df['AMT_ANNUITY'] ) 
    df['AMT_GOODS_PRICE']  =  np.log10(df['AMT_GOODS_PRICE'] ) 
    
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    '''
    for c in tqdm(cat_cols):
        gp = df.groupby(c)['AMT_ANNUITY']
        mean = gp.mean()
        std  = gp.std()
        df[c + '_deal_probability_avg'] = df[c].map(mean)
        df[c + '_deal_probability_std'] = df[c].map(std)
    '''        
    # Some simple new features (percentages)
    # df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    # df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    # df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    # df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df



#bureau = bureau[bureau['DAYS_CREDIT']>=-545]
# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True, limit=False): 
    bureau = pd.read_csv(PATH+'bureau.csv'+EXT, nrows = num_rows)
    if limit:
        bureau = bureau[bureau['DAYS_CREDIT']>=-limit]
    bureau.sort_values('DAYS_CREDIT',inplace=True)
    bb = pd.read_csv(PATH+'bureau_balance.csv'+EXT, nrows = num_rows)
    
    
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    
    
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [  'max', 'mean','sum', 'min','var','count'],
        'DAYS_CREDIT_ENDDATE': ['max', 'mean','sum', 'min','var','count'],
        'DAYS_CREDIT_UPDATE': ['max', 'mean','sum', 'min','var','count'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean','sum', 'min','var','count'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean','sum', 'min','var','count'],
        'AMT_CREDIT_SUM': ['max', 'mean','sum', 'min','var','count'],
        'AMT_CREDIT_SUM_DEBT': [ 'max', 'mean','sum', 'min','var','count'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean','sum', 'min','var','count'],
        'AMT_CREDIT_SUM_LIMIT': ['max', 'mean','sum', 'min','var','count'],
        'AMT_ANNUITY': ['max', 'mean','sum', 'min','var','count'],
        'CNT_CREDIT_PROLONG': ['max', 'mean','sum', 'min','var','count'],
        'MONTHS_BALANCE_MIN': ['max', 'mean','sum', 'min','var','count'],
        'MONTHS_BALANCE_MAX': ['max', 'mean','sum', 'min','var','count'],
        'MONTHS_BALANCE_SIZE': ['max', 'mean','sum', 'min','var','count']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau.sort_values('DAYS_CREDIT',inplace=True)
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    #num_aggregations['AMT_ANNUITY']=['max', 'mean']
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    bureau_agg.columns = [str(limit)+'_'+'bb_'+n for n in bureau_agg.columns]
    #print (bureau_agg.head())
    return bureau_agg




#prev[(prev['DAYS_FIRST_DUE']>-limit)&(prev['DAYS_FIRST_DUE']<365)]
# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True, limit=False):
    prev = pd.read_csv(PATH+'previous_application.csv'+EXT, nrows = num_rows)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    if limit:
        prev = prev[(prev['DAYS_FIRST_DUE']>-limit)]
    #print ('prev shape', prev.shape)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean','sum', 'min','var','count'],
        'AMT_APPLICATION': ['max', 'mean','sum', 'min','var','count'],
        'AMT_CREDIT': ['max', 'mean','sum', 'min','var','count'],
        'APP_CREDIT_PERC': ['max', 'mean','sum', 'min','var','count'],
        'AMT_DOWN_PAYMENT': ['max', 'mean','sum', 'min','var','count'],
        'AMT_GOODS_PRICE': ['max', 'mean','sum', 'min','var','count'],
        'HOUR_APPR_PROCESS_START': ['max', 'mean','sum', 'min','var','count'],
        'RATE_DOWN_PAYMENT': ['max', 'mean','sum', 'min','var','count'],
        'DAYS_DECISION': ['max', 'mean','sum', 'min','var','count'],
        'CNT_PAYMENT': ['max', 'mean','sum', 'min','var','count'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    
    if 'NAME_CONTRACT_STATUS_Approved' in prev.columns:
        print('adding approved')
        approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
        del approved, approved_agg
    else:
        print('no approved')        
    # Previous Applications: Refused Applications - only numerical features
    if 'NAME_CONTRACT_STATUS_Refused' in prev.columns:
        print('adding Refused')
        refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
        del refused, refused_agg
    else:
        print('no Refused')
    del prev
    gc.collect()
    prev_agg.columns = [str(limit)+'_'+'prevapp_'+n for n in prev_agg.columns]
    return prev_agg


#pos = pos[pos['MONTHS_BALANCE']>=-18]
# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True, limit=False):
    pos = pd.read_csv('input/POS_CASH_balance.csv.zip', nrows = num_rows)
    if limit:
        pos = pos[pos['MONTHS_BALANCE']>=-limit]
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean','sum', 'min','var','count'],
        'SK_DPD': ['max', 'mean','sum', 'min','var','count'],
        'SK_DPD_DEF': ['max', 'mean','sum', 'min','var','count'],
        'CNT_INSTALMENT_FUTURE':['max', 'mean','sum', 'min','var','count']
        
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    pos_agg.columns = [str(limit)+'_'+'poscash_'+n for n in pos_agg.columns]
    return pos_agg


#ins = ins[ins['DAYS_INSTALMENT']>=-730]
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True, limit=False):
    ins = pd.read_csv(PATH+'installments_payments.csv'+EXT, nrows = num_rows)
    if limit:
        ins = ins[ins['DAYS_INSTALMENT']>=-limit]
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique','count'],
        'DPD': ['max', 'mean','sum', 'min','var','count'],
        'DBD': ['max', 'mean','sum', 'min','var','count'],
        'PAYMENT_PERC': [ 'max', 'mean','sum', 'min','var','count'],
        'PAYMENT_DIFF': ['max', 'mean','sum', 'min','var','count'],
        'AMT_INSTALMENT': ['max', 'mean','sum', 'min','var','count'],
        'AMT_PAYMENT': ['max', 'mean','sum', 'min','var','count'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean','sum', 'min','var','count']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    ins_agg.columns = [str(limit)+'_'+'instpay_'+n for n in ins_agg.columns]
    return ins_agg

#cc = cc[cc['MONTHS_BALANCE']>=-18]
# Preprocess credit_card_balance.csv
    '''
        'AMT_BALANCE':['max','mean'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean'],
        'AMT_DRAWINGS_CURRENT': ['max', 'mean'], 
        'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'mean'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean'], 
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
        'AMT_PAYMENT_CURRENT': ['max', 'mean'], 
        'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean'],
        'AMT_RECEIVABLE_PRINCIPAL': ['max', 'mean'], 
        'AMT_RECIVABLE': ['max', 'mean'],
        'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['max', 'mean'], 
        'CNT_DRAWINGS_POS_CURRENT': ['max', 'mean'],
        'CNT_INSTALMENT_MATURE_CUM': ['max', 'mean'], 
        'NAME_CONTRACT_STATUS': ['max', 'mean']
        '''

def credit_card_balance(num_rows = None, nan_as_category = True, limit=False):
    print ('start')
    cc = pd.read_csv(PATH+'credit_card_balance.csv'+EXT , nrows = num_rows)
    if limit:
        cc = cc[cc['MONTHS_BALANCE']>=-limit]
    print ('cc shape:', cc.shape)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    #print (cc_agg.head())
    cc_agg.columns = [str(limit)+'_'+'credditcard_'+n for n in cc_agg.columns]
    print('__________')
    return cc_agg


'''
def credit_card_balance2(num_rows = None, nan_as_category = True):
    cc = pd.read_csv(PATH+'credit_card_balance.csv'+EXT , nrows = num_rows)
    grp = cc.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index().rename(index = str, columns = {'CNT_INSTALMENT_MATURE_CUM': 'NO_INSTALMENTS'})
    grp1 = grp.groupby(by = ['SK_ID_CURR'])['NO_INSTALMENTS'].sum().reset_index().rename(index = str, columns = {'NO_INSTALMENTS': 'TOTAL_INSTALMENTS'})
    grp1.columns = pd.Index(['SK_ID_CURR']+['CC2_' + e[0] + "_" + e[1].upper() for e in grp1.columns.tolist()[1:]])
    grp1.set_index('SK_ID_CURR',inplace=True)
    #print (grp1.head())
    return grp1



def credit_card_balance3(num_rows = None, nan_as_category = True):
    def f(DPD):
        # DPD is a series of values of SK_DPD for each of the groupby combination 
        # We convert it to a list to get the number of SK_DPD values NOT EQUALS ZERO
        x = DPD.tolist()
        c = 0
        for i,j in enumerate(x):
            if j != 0:
                c += 1
        return c 
    
    cc = pd.read_csv(PATH+'credit_card_balance.csv'+EXT , nrows = num_rows)
    grp = cc.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV']).apply(lambda x: f(x.SK_DPD)).reset_index().rename(index = str, columns = {0: 'NO_DPD'})
    grp1 = grp.groupby(by = ['SK_ID_CURR'])['NO_DPD'].mean().reset_index().rename(index = str, columns = {'NO_DPD' : 'DPD_COUNT'})
    grp1.columns = pd.Index(['SK_ID_CURR']+['CC3_' + e[0] + "_" + e[1].upper() for e in grp1.columns.tolist()[1:]])
    grp1.set_index('SK_ID_CURR',inplace=True)

    print (grp1.head())
    return grp1


def credit_card_balance4(num_rows = None, nan_as_category = True):
    def f(min_pay, total_pay):
        
        M = min_pay.tolist()
        T = total_pay.tolist()
        P = len(M)
        c = 0 
        # Find the count of transactions when Payment made is less than Minimum Payment 
        for i in range(len(M)):
            if T[i] < M[i]:
                c += 1  
        return (100*c)/P
   
    cc = pd.read_csv(PATH+'credit_card_balance.csv'+EXT , nrows = num_rows)
    grp = cc.groupby(by = ['SK_ID_CURR']).apply(lambda x: f(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT)).reset_index().rename(index = str, columns = { 0 : 'PERCENTAGE_MISSED_PAYMENTS'})
    print (grp.head())
    grp.columns = pd.Index(['SK_ID_CURR']+['CC3_' + e[0] + "_" + e[1].upper() for e in grp.columns.tolist()[1:]])
    grp.set_index('SK_ID_CURR',inplace=True)

    #print (grp1.head())
    return grp
'''




# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(train_df, test_df, num_folds, lr = 0.02, stratified = False, 
                   debug= False, log=False, submission_file_name=''):    
    # Divide in training/validation and test data

    
    #print (test_df.head())
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    #del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['SK_ID_CURR', 'TARGET']]
    
    print ('len_feat', len(feats))
    feature_importance_df['f']=feats
    
    
    y = train_df['TARGET']

    train_df = train_df[feats]
    
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, y)):
        train_x, train_y = train_df.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], y.iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=10,
            n_estimators=100000,
            learning_rate=lr,
            num_leaves=32,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1)

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / float(folds.n_splits)
        print('inside', len(feats))
        #fold_importance_df = pd.DataFrame()
        #fold_importance_df["feature"] = feats
        #fold_importance_df["importance"] = clf.feature_importances_
        print ('len clf.feature_importances_', len(clf.feature_importances_))
        
        print ('clf.n_features_', clf.n_features_)
        print (clf)
        feature_importance_df["fold"+str(n_fold)] = clf.feature_importances_
        #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        temp_str = 'Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]))
        print (temp_str)
        log.write(temp_str+'\n')
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    temp_str = 'Full AUC score %.6f' % roc_auc_score(y.values, oof_preds)
    print (temp_str)
    log.write(temp_str+'\n')
    # Write submission file and plot feature importance
    #if not debug:
    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(OUT+submission_file_name, index= False)
    np.savetxt(OUT+submission_file_name+'oof_preds_lgb.txt', oof_preds)#, fmt='%d'
    #display_importances(feature_importance_df, submission_file_name)
    log.close()
    return feature_importance_df


# XGB GBDT with KFold or Stratified KFold
def kfold_xgb(train_df, test_df, num_folds, lr = 0.01, stratified = False, debug= False, log=False, submission_file_name=''):
    #print ('________________________________',num_folds)
    # Divide in training/validation and test data
    #train_df = df[df['TARGET'].notnull()]
    #test_df = df[df['TARGET'].isnull()]
    #print (test_df.head())
    print("Starting xgb. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    #del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=2018)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=2018)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    #print ('________________________________',len(folds))
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        #if n_fold == 0: # REmove for full K-fold run
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf = XGBClassifier(learning_rate =lr, n_estimators=10000, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', 
        nthread=4, scale_pos_weight=2, seed=27)

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)
        xgbfir.saveXgbFI(clf, TopK=300, MaxTrees=200, OutputXlsxFile=OUT+'fimportance_xgb_fold_'+str(n_fold)+'.xlsx')
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats])[:, 1]  / float(folds.n_splits) # - Uncomment for K-fold 

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        temp_str = 'Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]))
        print (temp_str)
        log.write(temp_str+'\n')
        
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
        #np.save("xgb_oof_preds_1", oof_preds)
        #np.save("xgb_sub_preds_1", sub_preds)

    # print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    temp_str = 'Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds)
    print (temp_str)
    log.write(temp_str+'\n')
    
    #if not debug:
    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(OUT+submission_file_name, index= False)
    np.savetxt(OUT+'oof_preds_xgb.txt', oof_preds)
    #display_importances(feature_importance_df)
    #return feature_importance_df
    #test_df['TARGET'] = sub_preds.clip(0, 1)
    #test_df = test_df[['SK_ID_CURR', 'TARGET']]
    display_importances(feature_importance_df, submission_file_name)
    log.close()
    return feature_importance_df


# XGB GBDT with KFold or Stratified KFold
def kfold_cat(train_df, test_df, num_folds, lr = 0.01, stratified = False, debug= False, log=False, submission_file_name=''):
    #print ('________________________________',num_folds)
    # Divide in training/validation and test data
    #train_df = df[df['TARGET'].notnull()]
    train_df = train_df.fillna(-999)
    #test_df = df[df['TARGET'].isnull()]
    test_df = test_df.fillna(-999)
    #print (test_df.head())
    print("Starting cat. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    #del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=2018)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=2018)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    #print ('________________________________',len(folds))
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        #if n_fold == 0: # REmove for full K-fold run
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]


        cb_params = {
        'iterations':5000,
        'learning_rate':lr,
        'depth':6,
        'l2_leaf_reg':40,
        'bootstrap_type':'Bernoulli',
        'subsample':0.7,
        'scale_pos_weight':5,
        'eval_metric':'AUC',
        'metric_period':50,
        'od_type':'Iter',
        'od_wait':45,
        'allow_writing_files':False ,
        
        }


        #clf = XGBClassifier(learning_rate =lr, n_estimators=10000, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', 
        #nthread=4, scale_pos_weight=2, seed=27)
        
        clf=CatBoostClassifier(**cb_params)

        clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose= 100, use_best_model=True)
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats])[:, 1]  / float(folds.n_splits) # - Uncomment for K-fold 

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        temp_str = 'Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]))
        print (temp_str)
        log.write(temp_str+'\n')
        
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
        #np.save("xgb_oof_preds_1", oof_preds)
        #np.save("xgb_sub_preds_1", sub_preds)

    # print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    temp_str = 'Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds)
    print (temp_str)
    log.write(temp_str+'\n')
    
    #if not debug:
    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(OUT+submission_file_name, index= False)
    np.savetxt(OUT+'oof_preds_cat.txt', oof_preds)
    #display_importances(feature_importance_df)
    #return feature_importance_df
    #test_df['TARGET'] = sub_preds.clip(0, 1)
    #test_df = test_df[['SK_ID_CURR', 'TARGET']]
    #display_importances(feature_importance_df, submission_file_name)
    log.close()
    return feature_importance_df



# Display/plot feature importance
def display_importances(feature_importance_df_, fname):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(OUT+fname+'.png')



num_rows = None
debug=False
# if debug else None
df = application_train_test(num_rows)
df = reduce_mem_usage(df)

'''
with timer("Process credit card balance 3"):        
    cc = credit_card_balance(num_rows, limit=3)
    print("Credit card balance df shape:", cc.shape)
    cc = reduce_mem_usage(cc)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    temp_1 = cc.columns
    del cc
    gc.collect()
    
            
with timer("Process credit card balance"):        
    cc = credit_card_balance(num_rows, limit=False)
    print("Credit card balance df shape:", cc.shape)
    cc = reduce_mem_usage(cc)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    temp_2 = cc.columns
    del cc
    gc.collect()
                 

temp_1 = set( [ '_'.join(n.split('_')[1:]) for n in temp_1]  )
temp_2 = set( [ '_'.join(n.split('_')[1:]) for n in temp_2]  )
common = temp_1 & temp_2
'''
'''
for item in list(common):
    df['fc_'+item]=df['3_'+item]/df['False_'+item]
    
#df.drop(['3_'+item for item in list(common)],axis=1,inplace=True)


with timer("Process previous_applications 30"):
    prev = previous_applications(num_rows, limit=90)
    print("Previous applications df shape:", prev.shape)
    prev = reduce_mem_usage(prev)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    
    temp_1 = prev.columns
    del prev
'''
'''
with timer("Process previous_applications 90"):
    prev = previous_applications(num_rows, limit=90)
    print("Previous applications df shape:", prev.shape)
    prev = reduce_mem_usage(prev)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    temp_1 = prev.columns
    del prev
    gc.collect()  
''' 
'''           
with timer("Process previous_applications 90"):
    prev = previous_applications(num_rows, limit=False)
    print("Previous applications df shape:", prev.shape)
    prev = reduce_mem_usage(prev)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    temp_2 = prev.columns
    del prev 
    gc.collect()           


temp_1 = set( [ '_'.join(n.split('_')[1:]) for n in temp_1]  )
temp_2 = set( [ '_'.join(n.split('_')[1:]) for n in temp_2]  )
common = list(temp_1 & temp_2)
for item in common:
    df['fc_'+item]=df['90_'+item]/df['False_'+item]     
#for item in common:
    #df['fc_'+item]=df['90_'+item]/df['360_'+item]     

'''
#df.drop(['90_'+item for item in list(common)],axis=1,inplace=True)
#df.drop(['360_'+item for item in list(common)],axis=1,inplace=True)


'''
with timer("Process bureau and bureau_balance 90"):
    bureau = bureau_and_balance(num_rows, limit=90)
    print("Bureau df shape:", bureau.shape)
    temp_1 = bureau.columns
    bureau = reduce_mem_usage(bureau)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()
 
   
with timer("Process bureau and bureau_balance"):
    bureau = bureau_and_balance(num_rows, limit=False)
    print("Bureau df shape:", bureau.shape)
    bureau = reduce_mem_usage(bureau)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    temp_2 = bureau.columns
    del bureau
    gc.collect()
   

temp_1 = set( [ '_'.join(n.split('_')[1:]) for n in temp_1]  )
temp_2 = set( [ '_'.join(n.split('_')[1:]) for n in temp_2]  )
common = list(temp_1 & temp_2)
for item in common:
    df['fc_'+item]=df['90_'+item]/df['False_'+item]  
#df.drop(['90_'+item for item in list(common)],axis=1,inplace=True)
#df.drop(['360_'+item for item in list(common)],axis=1,inplace=True)
'''
'''
with timer("Process POS-CASH balance 1"):
    pos = pos_cash(num_rows, limit=False)
    print("Pos-cash balance df shape:", pos.shape)
    pos = reduce_mem_usage(pos)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    temp_1 = pos.columns
    del pos
    gc.collect()
'''
'''   
with timer("Process POS-CASH balance 3"):
    pos = pos_cash(num_rows, limit=3)
    print("Pos-cash balance df shape:", pos.shape)
    pos = reduce_mem_usage(pos)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    temp_1 = pos.columns
    del pos
    gc.collect()
    
with timer("Process POS-CASH balance "):
    pos = pos_cash(num_rows, limit=False)
    print("Pos-cash balance df shape:", pos.shape)
    pos = reduce_mem_usage(pos)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    temp_2 = pos.columns
    del pos
    gc.collect()

temp_1 = set( [ '_'.join(n.split('_')[1:]) for n in temp_1]  )
temp_2 = set( [ '_'.join(n.split('_')[1:]) for n in temp_2]  )
common = list(temp_1 & temp_2)
for item in common:
    df['fc_'+item]=df['3_'+item]/df['False_'+item]  

#df.drop(['3_'+item for item in list(common)],axis=1,inplace=True)
#df.drop(['12_'+item for item in list(common)],axis=1,inplace=True)

    
with timer("Process installments payments 90"):
    ins = installments_payments(num_rows,limit=90)
    print("Installments payments df shape:", ins.shape)
    ins = reduce_mem_usage(ins)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    temp_1 = ins.columns
    del ins
    gc.collect()


with timer("Process installments payments"):
    ins = installments_payments(num_rows,limit=False)
    print("Installments payments df shape:", ins.shape)
    ins = reduce_mem_usage(ins)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    temp_2 = ins.columns
    del ins
    gc.collect()         


temp_1 = set( [ '_'.join(n.split('_')[1:]) for n in temp_1]  )
temp_2 = set( [ '_'.join(n.split('_')[1:]) for n in temp_2]  )
common = list(temp_1 & temp_2)
for item in common:
    df['fc_'+item]=df['90_'+item]/df['False_'+item]
   
#df.drop(['90_'+item for item in list(common)],axis=1,inplace=True)
#df.drop(['360_'+item for item in list(common)],axis=1,inplace=True)
#df.drop(['90_'+item for item in list(common)],axis=1,inplace=True)
'''
with timer("Process credit card balance"):        
    cc = credit_card_balance(num_rows, limit=12)
    print("Credit card balance df shape:", cc.shape)
    cc = reduce_mem_usage(cc)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    temp_2 = cc.columns
    del cc
    gc.collect()

with timer("Process POS-CASH balance "):
    pos = pos_cash(num_rows, limit=12)
    print("Pos-cash balance df shape:", pos.shape)
    pos = reduce_mem_usage(pos)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    temp_2 = pos.columns
    del pos
    gc.collect()
with timer("Process installments payments"):
    ins = installments_payments(num_rows,limit=360)
    print("Installments payments df shape:", ins.shape)
    ins = reduce_mem_usage(ins)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    temp_2 = ins.columns
    del ins
    gc.collect()         
with timer("Process previous_applications 30"):
    prev = previous_applications(num_rows, limit=360)
    print("Previous applications df shape:", prev.shape)
    prev = reduce_mem_usage(prev)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    temp_1 = prev.columns
    del prev

with timer("Process bureau and bureau_balance 90"):
    bureau = bureau_and_balance(num_rows, limit=360)
    print("Bureau df shape:", bureau.shape)
    temp_1 = bureau.columns
    bureau = reduce_mem_usage(bureau)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()
    
df['f_12'] = df['AMT_GOODS_PRICE'] / df['AMT_ANNUITY']
#df['f_4'] = df['AMT_GOODS_PRICE']           / (df['INSTAL_AMT_PAYMENT_SUM']+1)
df['f_13'] = df['AMT_GOODS_PRICE'] / df['DAYS_EMPLOYED']
#df['f_14'] = df['AMT_GOODS_PRICE'] /   (df['BURO_CREDIT_ACTIVE_Active_MEAN']+1)
df['f_15'] =    (df['DAYS_BIRTH']+abs(df['DAYS_BIRTH'].min())) /   (df['DAYS_REGISTRATION']+abs(df['DAYS_REGISTRATION'].min()))
df['f_16'] = (df['DAYS_REGISTRATION']+abs(df['DAYS_REGISTRATION'].min()))  / (df['DAYS_EMPLOYED']+abs(df['DAYS_EMPLOYED'].min())) 
df['f_17'] =    (df['DAYS_BIRTH']+abs(df['DAYS_BIRTH'].min())) /   (df['DAYS_ID_PUBLISH']+abs(df['DAYS_ID_PUBLISH'].min()))
'''
df['f_18'] = df['6_POS_CNT_INSTALMENT_FUTURE_MAX']/df['POS_CNT_INSTALMENT_FUTURE_MAX']
df['f_19'] = df['6_INSTAL_DPD_MEAN']/df['INSTAL_DPD_MEAN']
df['f_20'] = df['6_INSTAL_AMT_PAYMENT_MIN']/df['INSTAL_AMT_PAYMENT_MIN']
df['f_21'] = df['6_INSTAL_DBD_SUM']/df['INSTAL_DBD_SUM']
df['f_22'] = df['180_BURO_DAYS_CREDIT_MEAN']/df['BURO_DAYS_CREDIT_MEAN']

df['f_23'] = df['6_POS_CNT_INSTALMENT_FUTURE_MAX']/df['12_POS_CNT_INSTALMENT_FUTURE_MAX']
df['f_24'] = df['6_INSTAL_DPD_MEAN']/df['12_INSTAL_DPD_MEAN']
df['f_25'] = df['6_INSTAL_AMT_PAYMENT_MIN']/df['12_INSTAL_AMT_PAYMENT_MIN']
df['f_26'] = df['6_INSTAL_DBD_SUM']/df['12_INSTAL_DBD_SUM']
df['f_27'] = df['180_BURO_DAYS_CREDIT_MEAN']/df['365_BURO_DAYS_CREDIT_MEAN']

df['f_28'] = df['6_POS_CNT_INSTALMENT_FUTURE_MAX']/df['POS_CNT_INSTALMENT_FUTURE_MAX']
df['f_29'] = df['18_INSTAL_DPD_MEAN']/df['INSTAL_DPD_MEAN']
df['f_30'] = df['18_INSTAL_AMT_PAYMENT_MIN']/df['INSTAL_AMT_PAYMENT_MIN']
df['f_31'] = df['18_INSTAL_DBD_SUM']/df['INSTAL_DBD_SUM']
df['f_32'] = df['545_BURO_DAYS_CREDIT_MEAN']/df['BURO_DAYS_CREDIT_MEAN']

df['f_32'] = df['12_POS_CNT_INSTALMENT_FUTURE_MAX']/df['POS_CNT_INSTALMENT_FUTURE_MAX']
df['f_33'] = df['12_INSTAL_DPD_MEAN']/df['INSTAL_DPD_MEAN']
df['f_34'] = df['12_INSTAL_AMT_PAYMENT_MIN']/df['INSTAL_AMT_PAYMENT_MIN']
df['f_35'] = df['12_INSTAL_DBD_SUM']/df['INSTAL_DBD_SUM']
df['f_36'] = df['365_BURO_DAYS_CREDIT_MEAN']/df['BURO_DAYS_CREDIT_MEAN']
'''



df['f70'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE']        
df['f71'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE']
df['f76'] = df['f70']*df['f71']
df['f72']=df['NEW_CREDIT_TO_ANNUITY_RATIO']/df['CODE_GENDER']
df['f73']=df['NEW_CREDIT_TO_ANNUITY_RATIO']/df['REGION_RATING_CLIENT_W_CITY']
df['f74']=df['DAYS_BIRTH']/df['EXT_SOURCE_1']
df['f75']=df['NEW_CREDIT_TO_ANNUITY_RATIO']/df['DEF_30_CNT_SOCIAL_CIRCLE']
#df['f76']=df['ACTIVE_AMT_CREDIT_SUM_MEAN']/df['ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN']
df['f88']=df['NEW_CREDIT_TO_ANNUITY_RATIO']/df['f71']
#df['f88']=df['NEW_CREDIT_TO_ANNUITY_RATIO']/df['NEW_EXT_SOURCES_MEAN']
#fimp = pd.read_table('out_7f/fimp_1_trend.csv',sep=',')
#fimp = fimp[fimp['importance']>6]

#to_use = [n for n in to_use if 'False' not in n]
#to_use+=[n+'instpay_INSTAL_PAYMENT_DIFF_COUNT' for n in ['90_', '180_', '270_', '360_', 'False_']]
#,'NEW_EXT_SOURCES_MEANp']#,'EXT_SOURCE_3p_m']
#print (to_use)
#Full AUC score 0.795304        
to_use = [n for n in df.columns if n not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]


#selected = open('feat_selected.txt').read().split('\n')
#to_use = [n for n in 
df =df[to_use+['SK_ID_CURR','TARGET']]
#df =reduce_mem_usage(df)
#df.to_pickle(OUT+'df.zip')  

#df.to_pickle(OUT+'df.zip')  
gc.collect()


print('__________')

temp_kris_1 = pd.read_csv('kris/lightGBM_test_predictions_rank_mean.csv')
temp_kris_1.set_index('SK_ID_CURR',inplace=True)

temp_kris_2 = pd.read_csv('kris/oof_pred.csv')
temp_kris_2.columns = ['SK_ID_CURR','TARGET']
temp_kris_2.set_index('SK_ID_CURR',inplace=True)

temp_kris_all = pd.concat([temp_kris_1,temp_kris_2])
temp_kris_all.columns = ['kris']

df.set_index('SK_ID_CURR',inplace=True)
df=df.join(temp_kris_all)


temp_kris_1 = pd.read_csv('kris/NN_test_pred.csv')
temp_kris_1.columns = ['SK_ID_CURR','TARGET']
temp_kris_1.set_index('SK_ID_CURR',inplace=True)

temp_kris_2 = pd.read_csv('kris/NN_oof_pred.csv')
temp_kris_2.columns = ['SK_ID_CURR','TARGET']
temp_kris_2.set_index('SK_ID_CURR',inplace=True)

temp_kris_all = pd.concat([temp_kris_1,temp_kris_2])
temp_kris_all.columns = ['kris_nn']

#df.set_index('SK_ID_CURR',inplace=True)
df=df.join(temp_kris_all)






neptune_1 = pd.read_csv('neptune_5/lightGBM_out_of_fold_train_predictions.csv')
neptune_1.set_index('SK_ID_CURR',inplace=True)
neptune_1.columns = ['fold_id', 'TARGET']
neptune_1.drop('fold_id', axis=1,inplace=True)


neptune_2 = pd.read_csv('neptune_5/lightGBM_out_of_fold_test_predictions.csv')
neptune_2 = neptune_2.groupby('SK_ID_CURR').agg('mean')
neptune_2.columns = ['fold_id', 'TARGET']
neptune_2.drop('fold_id', axis=1,inplace=True)
neptune_all = pd.concat([neptune_1,neptune_2])
neptune_all.columns = ['nept']
df=df.join(neptune_all)




temp = np.loadtxt('out_7f_fimp_10/submission_2.csvoof_preds_xgb.txt')
xgb_1 = pd.DataFrame()
xgb_1['SK_ID_CURR']=df[df['TARGET'].notnull()].index.values
xgb_1['TARGET']=temp


xgb_2 = pd.read_csv('out_7f_fimp_10/submission_2.csv')
xgb_all = pd.concat([xgb_1,xgb_2])
xgb_all.columns = ['SK_ID_CURR', 'xgb']
xgb_all.set_index('SK_ID_CURR',inplace=True)
df=df.join(xgb_all)


temp = np.loadtxt('out_7f_fimp_10/submission_3.csvoof_preds_cat.txt')
xgb_1 = pd.DataFrame()
xgb_1['SK_ID_CURR']=df[df['TARGET'].notnull()].index.values
xgb_1['TARGET']=temp

xgb_2 = pd.read_csv('out_7f_fimp_10/submission_3.csv')
xgb_all = pd.concat([xgb_1,xgb_2])
xgb_all.columns = ['SK_ID_CURR', 'cat']
xgb_all.set_index('SK_ID_CURR',inplace=True)
df=df.join(xgb_all)


'''
temp = np.loadtxt('out_7f_fimp_10_down/submission_1.csvoof_preds_lgb.txt')
xgb_1 = pd.DataFrame()
xgb_1['SK_ID_CURR']=df[df['TARGET'].notnull()].index.values
xgb_1['TARGET']=temp

xgb_2 = pd.read_csv('out_7f_fimp_10_down/submission_1.csv')
xgb_all = pd.concat([xgb_1,xgb_2])
xgb_all.columns = ['SK_ID_CURR', 'down']
xgb_all.set_index('SK_ID_CURR',inplace=True)
df=df.join(xgb_all)
'''
#[974]   training's auc: 0.832236        valid_1's auc: 0.796665
#[616]   training's auc: 0.823459        valid_1's auc: 0.80123
df.reset_index(inplace=True)
train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]

train_df['TARGET']=train_df['TARGET'].sample(frac=1).values
del df
gc.collect()
#df=reduce_mem_usage(df)
submission_file_name="submission_1_trend_fselection_all.csv"
feat_importance = kfold_lightgbm(train_df, test_df, num_folds= NUM_FOLDS, lr=0.01, 
                                 stratified= STRATIFIED, debug= debug, 
                                 log = open(OUT+'log_1_trend_fselection_all.txt','w'), 
                                 submission_file_name = submission_file_name)
                                 
#feat_importance = feat_importance[["feature", "importance"]].groupby("feature").mean()
feat_importance.to_csv(OUT+'fimp_all.csv') 


    


submission_file_name="submission_2.csv"
feat_importance = kfold_xgb(train_df, test_df, num_folds= NUM_FOLDS, lr=0.01, 
                                   stratified= STRATIFIED, debug= debug,
                                   log = open(OUT+'log_2.txt','w'), 
                                   submission_file_name = submission_file_name     )
                                 
feat_importance = feat_importance[["feature", "importance"]].groupby("feature").mean()
feat_importance.to_csv(OUT+'fimp_2.csv')




submission_file_name="submission_3.csv"
feat_importance = kfold_cat(train_df, test_df, num_folds= NUM_FOLDS, lr=0.01, 
                                   stratified= STRATIFIED, debug= debug,
                                   log = open(OUT+'log_3.txt','w'), 
                                   submission_file_name = submission_file_name )
                                 
feat_importance = feat_importance[["feature", "importance"]].groupby("feature").mean()
feat_importance.to_csv(OUT+'fimp_3.csv')




