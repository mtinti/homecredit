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

warnings.simplefilter(action='ignore', category=FutureWarning)
#PATH = '../input/'
#EXT = ''


PATH = 'input/'
EXT = '.zip'
NUM_FOLDS = 5
STRATIFIED = False
OUT = 'out_7f_fimp_10/'

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
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean','count'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
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
    if limit:
        prev[(prev['DAYS_FIRST_DUE']>-limit)&(prev['DAYS_FIRST_DUE']<365)]
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
        'AMT_ANNUITY': [ 'max', 'mean','sum'],
        'AMT_APPLICATION': ['min', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean','sum'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
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
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'CNT_INSTALMENT_FUTURE':['max']
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
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': [ 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': [ 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
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
def credit_card_balance(num_rows = None, nan_as_category = True, limit=False):
    cc = pd.read_csv(PATH+'credit_card_balance.csv'+EXT , nrows = num_rows)
    if limit:
        cc = cc[cc['MONTHS_BALANCE']>=-limit]
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
def kfold_lightgbm(df, num_folds, lr = 0.02, stratified = False, debug= False, log=False, submission_file_name=''):    
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    #print (test_df.head())
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
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
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
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

    temp_str = 'Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds)
    print (temp_str)
    log.write(temp_str+'\n')
    # Write submission file and plot feature importance
    #if not debug:
    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(OUT+submission_file_name, index= False)
    np.savetxt(OUT+submission_file_name+'oof_preds_lgb.txt', oof_preds)#, fmt='%d'
    display_importances(feature_importance_df, submission_file_name)
    log.close()
    return feature_importance_df


# XGB GBDT with KFold or Stratified KFold
def kfold_xgb(df, num_folds, lr = 0.01, stratified = False, debug= False, log=False, submission_file_name=''):
    #print ('________________________________',num_folds)
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
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
    np.savetxt(OUT+submission_file_name+'oof_preds_xgb.txt', oof_preds)
    #display_importances(feature_importance_df)
    #return feature_importance_df
    #test_df['TARGET'] = sub_preds.clip(0, 1)
    #test_df = test_df[['SK_ID_CURR', 'TARGET']]
    display_importances(feature_importance_df, submission_file_name)
    log.close()
    return feature_importance_df


# XGB GBDT with KFold or Stratified KFold
def kfold_cat(df, num_folds, lr = 0.01, stratified = False, debug= False, log=False, submission_file_name=''):
    #print ('________________________________',num_folds)
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    train_df = train_df.fillna(-999)
    test_df = df[df['TARGET'].isnull()]
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
    np.savetxt(OUT+submission_file_name+'oof_preds_cat.txt', oof_preds)
    #display_importances(feature_importance_df)
    #return feature_importance_df
    #test_df['TARGET'] = sub_preds.clip(0, 1)
    #test_df = test_df[['SK_ID_CURR', 'TARGET']]
    display_importances(feature_importance_df, submission_file_name)
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


def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()  
    with timer("Process bureau and bureau_balance 90"):
        bureau = bureau_and_balance(num_rows, limit=90)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()          
    with timer("Process bureau and bureau_balance 180"):
        bureau = bureau_and_balance(num_rows, limit=180)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()  
    with timer("Process bureau and bureau_balance 270"):
        bureau = bureau_and_balance(num_rows, limit=270)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect() 
    with timer("Process bureau and bureau_balance 360"):
        bureau = bureau_and_balance(num_rows, limit=360)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()         
#     with timer("Process bureau and bureau_balance 540"):
#         bureau = bureau_and_balance(num_rows, limit=450)
#         print("Bureau df shape:", bureau.shape)
#         df = df.join(bureau, how='left', on='SK_ID_CURR')
#         del bureau
#         gc.collect() 
#     with timer("Process bureau and bureau_balance 540"):
#         bureau = bureau_and_balance(num_rows, limit=540)
#         print("Bureau df shape:", bureau.shape)
#         df = df.join(bureau, how='left', on='SK_ID_CURR')
#         del bureau
#         gc.collect() 
         
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()  
    with timer("Process previous_applications 90"):
        prev = previous_applications(num_rows, limit=90)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev                 
    with timer("Process previous_applications 180"):
        prev = previous_applications(num_rows, limit=180)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev  
    with timer("Process previous_applications 270"):
        prev = previous_applications(num_rows, limit=270)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev         
    with timer("Process previous_applications 360"):
        prev = previous_applications(num_rows, limit=360)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev        
#     with timer("Process previous_applications 450"):
#         prev = previous_applications(num_rows, limit=450)
#         print("Previous applications df shape:", prev.shape)
#         df = df.join(prev, how='left', on='SK_ID_CURR')
#         del prev
#     with timer("Process previous_applications 540"):
#         prev = previous_applications(num_rows, limit=540)
#         print("Previous applications df shape:", prev.shape)
#         df = df.join(prev, how='left', on='SK_ID_CURR')
#         del prev        
# =============================================================================
        
   
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process POS-CASH balance 3"):
        pos = pos_cash(num_rows, limit=3)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()        
    with timer("Process POS-CASH balance 6"):
        pos = pos_cash(num_rows, limit=6)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()  
    with timer("Process POS-CASH balance 9"):
        pos = pos_cash(num_rows, limit=9)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process POS-CASH balance 12"):
        pos = pos_cash(num_rows, limit=12)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
#         gc.collect()
#     with timer("Process POS-CASH balance 15"):
#         pos = pos_cash(num_rows, limit=15)
#         print("Pos-cash balance df shape:", pos.shape)
#         df = df.join(pos, how='left', on='SK_ID_CURR')
#         del pos
#         gc.collect()
#     with timer("Process POS-CASH balance 15"):
#         pos = pos_cash(num_rows, limit=18)
#         print("Pos-cash balance df shape:", pos.shape)
#         df = df.join(pos, how='left', on='SK_ID_CURR')
#         del pos
#         gc.collect()
#     with timer("Process POS-CASH balance 15"):
#         pos = pos_cash(num_rows, limit=21)
#         print("Pos-cash balance df shape:", pos.shape)
#         df = df.join(pos, how='left', on='SK_ID_CURR')
#         del pos
#         gc.collect()
#     with timer("Process POS-CASH balance 15"):
#         pos = pos_cash(num_rows, limit=24)
#         print("Pos-cash balance df shape:", pos.shape)
#         df = df.join(pos, how='left', on='SK_ID_CURR')
#         del pos
#         gc.collect()
# =============================================================================
        
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process installments payments 290"):
        ins = installments_payments(num_rows,limit=90)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()         
    with timer("Process installments payments 180"):
        ins = installments_payments(num_rows,limit=180)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process installments payments 270"):
        ins = installments_payments(num_rows, limit=270)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()         
    with timer("Process installments payments 360"):
        ins = installments_payments(num_rows, limit=360)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()  
#     with timer("Process installments payments 540"):
#         ins = installments_payments(num_rows, limit=450)
#         print("Installments payments df shape:", ins.shape)
#         df = df.join(ins, how='left', on='SK_ID_CURR')
#         del ins
#         gc.collect()  
#     with timer("Process installments payments 540"):
#         ins = installments_payments(num_rows, limit=540)
#         print("Installments payments df shape:", ins.shape)
#         df = df.join(ins, how='left', on='SK_ID_CURR')
#         del ins
#         gc.collect() 
# =============================================================================
        
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect() 
    with timer("Process credit card balance 3"):        
        cc = credit_card_balance(num_rows, limit=3)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()        
    with timer("Process credit card balance 6"):
        cc = credit_card_balance(num_rows, limit=6)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Process credit card balance 9"):
        cc = credit_card_balance(num_rows, limit=9)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()         
    with timer("Process credit card balance 12"):
        cc = credit_card_balance(num_rows, limit=12)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()        
#     with timer("Process credit card balance 16"):
#         cc = credit_card_balance(num_rows, limit=15)
#         print("Credit card balance df shape:", cc.shape)
#         df = df.join(cc, how='left', on='SK_ID_CURR')
#         del cc
#         gc.collect()         
#     with timer("Process credit card balance 18"):
#         cc = credit_card_balance(num_rows, limit=18)
#         print("Credit card balance df shape:", cc.shape)
#         df = df.join(cc, how='left', on='SK_ID_CURR')
#         del cc
#         gc.collect()         
#     with timer("Process credit card balance 21"):
#         cc = credit_card_balance(num_rows, limit=21)
#         print("Credit card balance df shape:", cc.shape)
#         df = df.join(cc, how='left', on='SK_ID_CURR')
#         del cc
#         gc.collect()    
#     with timer("Process credit card balance 24"):
#         cc = credit_card_balance(num_rows, limit=24)
#         print("Credit card balance df shape:", cc.shape)
#         df = df.join(cc, how='left', on='SK_ID_CURR')
#         del cc
#         gc.collect()  
#         
# =============================================================================
    with timer("Run LightGBM with kfold"):
        
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
        
        #the folloving columns taken from
        #https://www.kaggle.com/ogrellier/lighgbm-with-selected-features/code
        print('deleting columns')
        #str(limit)+'_'+'bb_'+n for n
        print (df.columns)
        not_important = open('input/no_importance_lgb.txt','r').read().split('\n')
        #to_delete = [n for n in df.columns if n in not_important]
        
        prefixes = ['90_','180_','270_','360_','3_','6_','9_','12_']
        prefixes_2 = ['bb_','prevapp_','poscash_','instpay_' ,'credditcard_']
        
        final_delete = set([a+b+c for a in prefixes for b in prefixes_2 for c in not_important])
        all_columns = set(df.columns)
        final_delete = final_delete & all_columns
        #print (final_delete)
        df.drop(list(final_delete), axis=1, inplace=True)
        
        
        fimp = pd.read_table('out_7f/fimp_1.csv',sep=',')
        fimp = fimp[fimp['importance']>10]
        to_use = list(fimp['feature'])
        #print (to_use)
        df =df[to_use+['SK_ID_CURR', 'TARGET']]
        
        #df['f_18'] = df['3_poscash_POS_CNT_INSTALMENT_FUTURE_MAX']/df['12_poscash_POS_CNT_INSTALMENT_FUTURE_MAX']
        #df['f_19'] = df['3_poscash_POS_CNT_INSTALMENT_FUTURE_MAX']+df['90_prevapp_PREV_CNT_PAYMENT_MEAN']
        #df['f_18'] = df['NEW_CREDIT_TO_ANNUITY_RATIO']*(df['3_poscash_POS_CNT_INSTALMENT_FUTURE_MAX'])#df['90_prevapp_PREV_CNT_PAYMENT_MEAN'])
        #df['f_19'] = df['AMT_ANNUITY'] / df['3_poscash_POS_CNT_INSTALMENT_FUTURE_MAX']
        #df['f_20']=  df['AMT_GOODS_PRICE']  /  df['3_poscash_POS_CNT_INSTALMENT_FUTURE_MAX']
        #df['f_21']=  df['AMT_INCOME_TOTAL']  /  df['3_poscash_POS_CNT_INSTALMENT_FUTURE_MAX']
        
        #df['f_22'] = df['AMT_CREDIT'] / df['90_prevapp_PREV_CNT_PAYMENT_MEAN']
        #df['f_23'] = df['AMT_ANNUITY'] /  df['90_prevapp_PREV_CNT_PAYMENT_MEAN']
        #df['f_24']=  df['AMT_GOODS_PRICE']  /  df['90_prevapp_PREV_CNT_PAYMENT_MEAN']
        #df['f_25']=  df['AMT_INCOME_TOTAL']  /  df['90_prevapp_PREV_CNT_PAYMENT_MEAN']
        
        
        #df['AMT_INCOME_TOTAL']
       

        '''
        for col in df.columns:
            if col in not_important:
                del df[col]
                if '90_'+col in df.columns:
                    del df['90_'+col]
                if '180_'+col in df.columns:
                    del df['180_'+col]
                if '270_'+col in df.columns:
                    del df['270_'+col]                    
                if '360_'+col in df.columns:
                    del df['360_'+col]
                if '450_'+col in df.columns:
                    del df['450_'+col]                    
                if '540_'+col in df.columns:
                    del df['540_'+col] 
                
                if '3_'+col in df.columns:
                    del df['3_'+col]                     
                if '6_'+col in df.columns:
                    del df['6_'+col] 
                if '9_'+col in df.columns:
                    del df['9_'+col]                    
                if '12_'+col in df.columns:
                    del df['12_'+col] 
                if '15_'+col in df.columns:
                    del df['15_'+col]                     
                if '18_'+col in df.columns:
                    del df['18_'+col] 
        '''
        '''            
        not_important = open('input/no_importance_7d.txt','r').read().split('\n')
        for col in df.columns:
            if col in not_important:
                del df[col]
        '''           
        gc.collect()
        
        
        print('__________')

        
        submission_file_name="submission_1.csv"
        feat_importance = kfold_lightgbm(df, num_folds= NUM_FOLDS, lr=0.01, 
                                         stratified= STRATIFIED, debug= debug, 
                                         log = open(OUT+'log_1.txt','w'), 
                                         submission_file_name = submission_file_name)
                                         
        feat_importance = feat_importance[["feature", "importance"]].groupby("feature").mean()
        feat_importance.to_csv(OUT+'fimp_1.csv') 
        

        
        submission_file_name="submission_2.csv"
        feat_importance = kfold_xgb(df, num_folds= NUM_FOLDS, lr=0.01, 
                                           stratified= STRATIFIED, debug= debug,
                                           log = open(OUT+'log_2.txt','w'), 
                                           submission_file_name = submission_file_name     )
                                         
        feat_importance = feat_importance[["feature", "importance"]].groupby("feature").mean()
        feat_importance.to_csv(OUT+'fimp_2.csv')
        
        
        
        
        submission_file_name="submission_3.csv"
        feat_importance = kfold_cat(df, num_folds= NUM_FOLDS, lr=0.01, 
                                           stratified= STRATIFIED, debug= debug,
                                           log = open(OUT+'log_3.txt','w'), 
                                           submission_file_name = submission_file_name )
                                         
        feat_importance = feat_importance[["feature", "importance"]].groupby("feature").mean()
        feat_importance.to_csv(OUT+'fimp_3.csv')
        


if __name__ == "__main__":
    #submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()