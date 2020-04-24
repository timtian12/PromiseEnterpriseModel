#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
from catboost import CatBoostClassifier

# # 加载数据

# In[2]:


train_data = pd.read_csv('./shandong_data/train_stage2_update_20200320.csv')
train_label = pd.read_csv('./shandong_data/train_label.csv')
train_data = train_data.merge(train_label, on='ID', how='left')

test_data = pd.read_csv('./shandong_data/test_stage2_update_20200320.csv')
test_data['Label']=-1


# In[3]:


# 删除无效数据
def get_data_null_rate(train, test, cols=None):
    if cols!=None:
        train = train[cols]
        test = test[cols]
    data_null_rate = (train.isnull().sum()/len(train)).reset_index().rename(columns={0:'train'})
    data_null_rate = data_null_rate.merge((test.isnull().sum()/len(test)).reset_index().rename(columns={0:'test'}),
                                          on='index',how='inner')
    del train,test
    gc.collect()
    return data_null_rate
data_null_rate = get_data_null_rate(train_data.copy(),test_data.copy())
# 删除训练集和验证集中缺失率为1的 特征
rm_cols = list(data_null_rate[(data_null_rate['train']==1)|(data_null_rate['test']==1)]['index'])
for col in train_data.columns:
    if train_data[col].nunique() < 2:
        rm_cols.append(col)

train_data = train_data[[col for col in train_data.columns if col not in rm_cols]]
test_data = test_data[[col for col in test_data.columns if col not in rm_cols]]
all_data = train_data.append(test_data)


# # 数据清洗过程

# In[4]:


# 处理时间
date_cols = ['经营期限自','经营期限至','成立日期','核准日期','注销时间']
all_data['date_null_rate'] = all_data[date_cols].apply(lambda x: x.isnull().sum(), axis=1)
# 分割出日期类型中得小时
def split_date(x):
#     print(x,type(x))
    if type(x)==float:
        return np.nan
    else:
        return int(x.split(':')[0])
for col in date_cols:
    all_data[col] = all_data[col].apply(lambda x: split_date(x))


# In[5]:


# 处理邮编
def youbian_encode(x):
    if len(re.match('[\d]*',str(x)).group())==0:
        x = 999999
    elif len(str(int(float(x))))!=6:
        if len(str(int(float(x))))>3:
            x = int(str(int(float(x)))+'0'*(6-len(str(int(float(x))))))
        elif len(str(int(float(x))))==3:
            x=int('266'+str(int(float(x))))
        elif len(str(int(float(x))))==2:
            x = int('2667'+str(int(float(x))))
        else:
            x = 266000
    elif pd.isna(x):
        x=np.nan
    else:
        x = int(float(x))
    return x
all_data['邮政编码'].fillna(999999,inplace=True)
all_data['邮政编码'] = all_data['邮政编码'].apply(lambda x: youbian_encode(x))


# In[6]:


# 处理离群值
def outliers_proc(data, col_name, scale=3):
    """
        用于截尾异常值， 默认用box_plot(scale=3)进行清洗
        param:
            data：接收pandas数据格式
            col_name: pandas列名
            scale: 尺度
    """
    data_col = data[col_name]
    Q1 = data_col.quantile(0.25) # 0.25分位数
    Q3 = data_col.quantile(0.75)  # 0,75分位数
    IQR = Q3 - Q1

    data_col[data_col < Q1 - (scale * IQR)] = Q1 - (scale * IQR)
    data_col[data_col > Q3 + (scale * IQR)] = Q3 + (scale * IQR)

    return data[col_name]
for col in ['注册资本','增值税','企业所得税','印花税','教育费','城建税','年度参保总额']:
    all_data[col] = outliers_proc(all_data, col, 10)


# In[7]:


# 经营范围
def process_jingyingfanwei(x):
    vals = [i.strip() for i in x[1:-1].split(',')]
    return ' '.join(vals)
all_data['经营范围'] = all_data['经营范围'].apply(lambda x: process_jingyingfanwei(x))


# # 特征工程

# In[8]:


def tf_idf(all_data, col, n_components=5):
    Tfidf_vect = TfidfVectorizer()
    tfidf_vec = Tfidf_vect.fit_transform(all_data[col])
    svd_enc = TruncatedSVD(n_components=n_components, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['SVD_{}_{}'.format(col, i) for i in range(n_components)]
    vec_svd = vec_svd.reset_index(drop= True)
    return vec_svd


# In[9]:


# 经营范围编码
jingying_svd = tf_idf(all_data, '经营范围', n_components=5)
all_data = all_data.reset_index(drop= True)
all_data = pd.concat([all_data, jingying_svd], axis=1)


# In[10]:


money_cols = ['投资总额','注册资本','增值税','企业所得税','印花税','教育费','城建税','年度参保总额']
all_data['教育费/增值税'] = all_data['教育费']/all_data['增值税']
all_data['教育费/企业所得'] = all_data['教育费']/all_data['企业所得税']
all_data['企业所得/教育费'] = all_data['企业所得税']/all_data['教育费']
all_data['企业所得税/城建税'] = all_data['企业所得税']/all_data['城建税']
all_data['城建税/教育费'] = all_data['城建税']/all_data['教育费']


# In[11]:


anjian_cols = [col for col in all_data.columns if col.find('案件')!=-1]
all_data['1_3_anjian'] = all_data[['1月案件数','2月案件数','3月案件数']].apply(lambda x: x.sum(),axis=1)
all_data['1_3_anjian'].replace(0,np.nan,inplace=True)
all_data['4_5_anjian'] = all_data[['4月案件数','5月案件数','6月案件数']].apply(lambda x: x.sum(),axis=1)
all_data['4_5_anjian'].replace(0,np.nan,inplace=True)
all_data['7_9_anjian'] = all_data[['7月案件数','8月案件数','9月案件数']].apply(lambda x: x.sum(),axis=1)
all_data['7_9_anjian'].replace(0,np.nan,inplace=True)
all_data['10_12_anjian'] = all_data[['10月案件数','11月案件数','12月案件数']].apply(lambda x: x.sum(),axis=1)
all_data['10_12_anjian'].replace(0,np.nan,inplace=True)


# In[12]:


zichan_cols = [col for col in all_data.columns if col.find('变更')!=-1]
all_data['zichan'] = (all_data['资本变更后']-all_data['资本变更前'])


# In[13]:


def process_rank(x):
    res = [str(int(i)) for i in list(x)]
    return  ' '.join(res)
cols2=[]
tmp = pd.DataFrame()
for col1 in ['投资总额','注册资本','增值税','企业所得税','印花税','教育费','城建税']:
    for col2 in ['邮政编码','行业代码','企业类型','行业门类']:
        tmp[col1+'_'+col2] = all_data[col1].groupby(all_data[col2]).rank(ascending=0,method='max')
tmp.fillna(-1,inplace=True)
tmp['rank'] = tmp.apply(lambda x:process_rank(x),axis=1)
# rank 编码
rank_svd = tf_idf(tmp, 'rank', n_components=3)
all_data = all_data.reset_index(drop= True)
all_data = pd.concat([all_data, rank_svd], axis=1)


# In[14]:


# 处理行业门类
def process_hangye(x):
    if x>12:
        return 12 
#     elif x==4:
#         return 3
all_data['行业门类'] = all_data['行业门类'].apply(lambda x: process_hangye(x))


# In[15]:


def train_lgb_model(train_, valid_, id_name, label_name, categorical_feature=None, seed=1024, is_shuffle=True):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_['res'] = 0
    pred = [col for col in train_.columns if col not in [id_name,label_name,'res']]
    sub_preds = np.zeros((valid_.shape[0], folds.n_splits))
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'num_leaves':32,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'seed': 1,
        # 'device': 'gpu',
        'bagging_seed': 1,
        'feature_fraction_seed':7,
        'min_data_in_leaf': 28,
        'nthread': -1,
        'verbose': -1, 
    }
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_, train_[label_name]), start=1):
        print(f'the {n_fold} training start ...')
        train_x, train_y = train_[pred].iloc[train_idx], train_[label_name].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx], train_[label_name].iloc[valid_idx]
        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y)
        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=100,
            verbose_eval=100,
        )
        train_pred = clf.predict(valid_x, num_iteration=clf.best_iteration)
        train_['res'].iloc[valid_idx] = train_['res'].iloc[valid_idx] + train_pred
        
        # 在验证集上
        sub_preds[:, n_fold - 1] = clf.predict(valid_[pred], num_iteration=clf.best_iteration)
    valid_[label_name] = np.mean(sub_preds, axis=1)

    return train_[[id_name, 'res']],valid_[[id_name,label_name]]


# In[16]:


def train_xgb_model(train_, valid_, id_name, label_name, categorical_feature=None, seed=1024, is_shuffle=True):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_['res'] = 0
    pred = [col for col in train_.columns if col not in [id_name,label_name,'res']]
    valid_dms = xgb.DMatrix(valid_[pred])
    sub_preds = np.zeros((valid_.shape[0], folds.n_splits))
    params = {'eta': 0.1,
              'tree_method': "hist",
              'grow_policy': "lossguide",
              'max_leaves': 1024,
              'max_depth': 7,
              'subsample': 0.9,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'alpha': 4,
              'objective': 'binary:logistic',
              'scale_pos_weight': 9,
              'eval_metric': ['auc'],
              'nthread': 8,
              'random_state': 99,
              'silent': True}
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_, train_[label_name]), start=1):
        train_x, train_y = train_[pred].iloc[train_idx], train_[label_name].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx], train_[label_name].iloc[valid_idx]
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        clf = xgb.train(params, dtrain, 2000, watchlist,
                          maximize=True, early_stopping_rounds=50, verbose_eval=10)
        train_pred = clf.predict(dvalid)
        train_['res'].iloc[valid_idx] = train_['res'].iloc[valid_idx] + train_pred
        sub_preds[:, n_fold - 1] = clf.predict(valid_dms)
    valid_[label_name] = np.mean(sub_preds, axis=1)

    return train_[[id_name, 'res']],valid_[[id_name,label_name]]


# In[17]:


def train_cat_model(train_, valid_, id_name, label_name, categorical_feature=None, seed=1024, is_shuffle=False):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_['res'] = 0
    pred = [col for col in train_.columns if col not in [id_name,label_name,'res']]
    sub_preds = np.zeros((valid_.shape[0], folds.n_splits))
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_, train_[label_name]), start=1):
        print(f'the {n_fold} training start ...')
        train_x, train_y = train_[pred].iloc[train_idx], train_[label_name].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx], train_[label_name].iloc[valid_idx]
        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y)
        clf = CatBoostClassifier(iterations=500, learning_rate=0.12, 
                                 loss_function='Logloss',
                                 eval_metric='AUC',
                                 verbose=100,
                                 depth=6,
                                 l2_leaf_reg=5,
                                 cat_features=categorical_feature)
        clf.fit(train_x,train_y,eval_set=(valid_x, valid_y),early_stopping_rounds=50)
        train_pred = clf.predict_proba(valid_x)[:,1]
        train_['res'].iloc[valid_idx] = train_['res'].iloc[valid_idx] + train_pred
        sub_preds[:, n_fold - 1] = clf.predict_proba(valid_[pred])[:,1]
        
    valid_[label_name] = np.mean(sub_preds, axis=1)

    return train_[[id_name, 'res']],valid_[[id_name,label_name]]


# In[18]:


cols = ['城建税/教育费', '印花税', '年度参保总额', '最新参保人数', '企业所得税', 'SVD_rank_1', 'SVD_rank_0', '行业代码', '增值税', '教育费/增值税', 'SVD_经营范围_4', 'SVD_rank_2', 'SVD_经营范围_0', '城建税', 'SVD_经营范围_3', '登记注册类型代码', '注册资本', '经营期限自', 'SVD_经营范围_1', '成立日期', 'SVD_经营范围_2', '企业类型', '企业所得税/城建税', '教育费', '教育费/企业所得', '邮政编码', '登记机关', '核准日期', '经营期限至', '管辖机关', '诉讼总数量', '企业所得/教育费', '货币资金_年初数', '投资总额', '货币资金_年末数', '纳税人状态代码', '7_9_anjian', 'zichan', '1_3_anjian', '其他应收款_年末数', '其他应收款_年初数', '应收账款_年初数', '未分配利润_年初数', '预付款项_年末数', '预付款项_年初数', 'date_null_rate', '4_5_anjian', '所有者权益合计_年初数', '企业状态', '应付账款_年初数', '实收资本（股本）_年初数']
train_ = all_data[all_data['Label']!=-1]
valid_ = all_data[all_data['Label']==-1]


# In[ ]:





# In[19]:


lgb_train_pred, lgb_test_pred = train_lgb_model(train_[cols+['ID','Label']], valid_[cols+['ID']], 'ID', 'Label',seed=1024)
xgb_train_pred, xgb_test_pred = train_xgb_model(train_[cols+['ID','Label']], valid_[cols+['ID','Label']], 'ID', 'Label',seed=1024)
cat_train_pred, cat_test_pred = train_cat_model(train_[cols+['ID','Label']], valid_[cols+['ID','Label']], 'ID', 'Label',seed=1024) 


# In[20]:


# 将预测结果作为特征，训练lgb模型
lgb_train_pred.rename(columns={'res':'lgb_pred'},inplace=True)
xgb_train_pred.rename(columns={'res':'xgb_pred'},inplace=True)
cat_train_pred.rename(columns={'res':'cab_pred'},inplace=True)

lgb_test_pred.rename(columns={'Label':'lgb_pred'},inplace=True)
xgb_test_pred.rename(columns={'Label':'xgb_pred'},inplace=True)
cat_test_pred.rename(columns={'Label':'cab_pred'},inplace=True)

stacking_train_data = lgb_train_pred.merge(xgb_train_pred.rename(columns={}),on='ID',how='left')
stacking_train_data = stacking_train_data.merge(cat_train_pred,on='ID',how='left')

stacking_test_data = lgb_test_pred.merge(xgb_test_pred.rename(columns={}),on='ID',how='left')
stacking_test_data = stacking_test_data.merge(cat_test_pred,on='ID',how='left')


train_2 = train_.merge(stacking_train_data, on='ID', how='left')
valid_2 = valid_.merge(stacking_test_data, on='ID', how='left')
lgb_train_pred_2, lgb_test_pred_2 = train_lgb_model(train_2[cols+['lgb_pred','xgb_pred','cab_pred']+['ID','Label']], 
                                                            valid_2[cols+['lgb_pred','xgb_pred','cab_pred']+['ID']], 
                                                            'ID', 'Label',seed=1024)


# In[21]:


lgb_test_pred_2.to_csv('./submit.csv',index=False)


# In[ ]:




