#!/usr/bin/env python
# coding: utf-8

# Credit to 此般浅薄 for the initial inspiration

# In[1]:


# Install tsflex and seglearn
get_ipython().system('pip install tsflex --no-index --find-links=file:///kaggle/input/time-series-tools')
get_ipython().system('pip install seglearn --no-index --find-links=file:///kaggle/input/time-series-tools')


# In[2]:


import numpy as np
import pandas as pd
from sklearn import *
import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
from pathlib import Path
from seglearn.feature_functions import base_features, emg_features
from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
from tsflex.features.integrations import seglearn_feature_dict_wrapper
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone
from sklearn.metrics import average_precision_score


# # Grab important files

# In[3]:


root = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/'

train = glob.glob(path.join(root, 'train/**/**'))
test = glob.glob(path.join(root, 'test/**/**'))

subjects = pd.read_csv(path.join(root, 'subjects.csv'))
tasks = pd.read_csv(path.join(root, 'tasks.csv'))
events = pd.read_csv(path.join(root, 'events.csv'))

tdcsfog_metadata = pd.read_csv(path.join(root, 'tdcsfog_metadata.csv'))
defog_metadata = pd.read_csv(path.join(root, 'defog_metadata.csv')) 

tdcsfog_metadata['Module'] = 'tdcsfog'
defog_metadata['Module'] = 'defog'

full_metadata = pd.concat([tdcsfog_metadata, defog_metadata])


# In[4]:


subjects.loc[subjects['Subject'] == 'fe5d84', 'Sex'] = 'F'


# In[5]:


seed = 100
cluster_size = 8


# In[6]:


subjects['Sex'] = subjects['Sex'].factorize()[0]
subjects = subjects.fillna(0).groupby('Subject').median()
subjects['s_group'] = cluster.KMeans(n_clusters = cluster_size, random_state = seed).fit_predict(subjects[subjects.columns[1:]])
new_names = {'Visit':'s_visit','Age':'s_age','YearsSinceDx':'s_years','UPDRSIII_On':'s_on','UPDRSIII_Off':'s_off','NFOGQ':'s_NFOGQ', 'Sex': 's_sex'}
subjects = subjects.rename(columns = new_names)
subjects


# In[7]:


print(tasks.keys())
tasks['Duration'] = tasks['End'] - tasks['Begin']
tasks = pd.pivot_table(tasks, values=['Duration'], index=['Id'], columns=['Task'], aggfunc='sum', fill_value=0)
tasks.columns = [c[1] for c in tasks.columns]
tasks = tasks.reset_index()
tasks['t_group'] = cluster.KMeans(n_clusters = cluster_size, random_state = seed).fit_predict(tasks[tasks.columns[1:]])


# In[8]:


# merge the subjects with the metadata
metadata_w_subjects = full_metadata.merge(subjects, how='left', on='Subject').copy()
features = metadata_w_subjects.columns


# In[9]:


metadata_w_subjects['Medication'] = metadata_w_subjects['Medication'].factorize()[0]


# # Extract from seglearn.feature_functions import base_features, emg_features
# 
# from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
# from tsflex.features.integrations import seglearn_feature_dict_wrapper from the time series data itself

# In[10]:


basic_feats = MultipleFeatureDescriptors(
    functions=seglearn_feature_dict_wrapper(base_features()),
    series_names=['AccV', 'AccML', 'AccAP'],
    windows=[10000],
    strides=[10000],
)

emg_feats = emg_features()
del emg_feats['simple square integral'] # is same as abs_energy (which is in base_features)

emg_feats = MultipleFeatureDescriptors(
    functions=seglearn_feature_dict_wrapper(emg_feats),
    series_names=['AccV', 'AccML', 'AccAP'],
    windows=[10000],
    strides=[10000],
)

fc = FeatureCollection([basic_feats, emg_feats])


# In[11]:


def reader(file):
    try:
        df = pd.read_csv(file, index_col='Time', usecols=['Time', 'AccV', 'AccML', 'AccAP', 'StartHesitation', 'Turn' , 'Walking'])

        path_split = file.split('/')
        df['Id'] = path_split[-1].split('.')[0]
        dataset = Path(file).parts[-2]
        df['Module'] = dataset
        
        # this is done because the speeds are at different rates for the datasets
#         if dataset == 'tdcsfog':
#             df.AccV = df.AccV / 9.80665
#             df.AccML = df.AccML / 9.80665
#             df.AccAP = df.AccAP / 9.80665

        df['Time_frac']=(df.index/df.index.max()).values
        
        df = pd.merge(df, tasks[['Id','t_group']], how='left', on='Id').fillna(-1)
        
        df = pd.merge(df, metadata_w_subjects[['Id','Subject', 'Visit','Test','Medication','s_group']], how='left', on='Id').fillna(-1)
        
        df_feats = fc.calculate(df, return_df=True, include_final_window=True, approve_sparsity=True, window_idx="begin").astype(np.float32)
        df = df.merge(df_feats, how="left", left_index=True, right_index=True)
        
#         # stride
#         df["Stride"] = df["AccV"] + df["AccML"] + df["AccAP"]

#         # step
#         df["Step"] = np.sqrt(abs(df["Stride"]))
    
        df.fillna(method="ffill", inplace=True)
        
        return df
    except: pass

train = pd.concat([reader(f) for f in tqdm(train)]).fillna(0); print(train.shape)
cols = [c for c in train.columns if c not in ['Id','Subject','Module', 'Time', 'StartHesitation', 'Turn' , 'Walking', 'Valid', 'Task','Event']]
pcols = ['StartHesitation', 'Turn' , 'Walking']
scols = ['Id', 'StartHesitation', 'Turn' , 'Walking']
train=train.reset_index(drop=True)


# In[12]:


train.head()


# In[13]:


best_params_ = {'colsample_bytree': 0.5282057895135501,
 'learning_rate': 0.22659963168004743,
 'max_depth': 8,
 'min_child_weight': 3.1233911067827616,
 'n_estimators': 291,
 'subsample': 0.9961057796456088,
 }

def custom_average_precision(y_true, y_pred):
    score = average_precision_score(y_true, y_pred)
    return 'average_precision', score, True

class LGBMMultiOutputRegressor(MultiOutputRegressor):
    def fit(self, X, y, eval_set=None, **fit_params):
        self.estimators_ = [clone(self.estimator) for _ in range(y.shape[1])]
        
        for i, estimator in enumerate(self.estimators_):
            if eval_set:
                fit_params['eval_set'] = [(eval_set[0], eval_set[1][:, i])]
            estimator.fit(X, y[:, i], **fit_params)
        
        return self


# In[14]:


kfold = GroupKFold(5)
groups=kfold.split(train, groups=train.Subject)

regs = []
cvs = []

for _, (tr_idx, te_idx) in enumerate(tqdm(groups, total=5, desc="Folds")):
    
    tr_idx = pd.Series(tr_idx).sample(n=2000000,random_state=42).values

    multioutput_regressor = LGBMMultiOutputRegressor(lgb.LGBMRegressor(**best_params_))

    x_train = train.loc[tr_idx, cols].to_numpy()
    y_train = train.loc[tr_idx, pcols].to_numpy()
    
    x_test = train.loc[te_idx, cols].to_numpy()
    y_test = train.loc[te_idx, pcols].to_numpy()

    multioutput_regressor.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        eval_metric=custom_average_precision,
        early_stopping_rounds=15,
        verbose = 0,
    )
    
    regs.append(multioutput_regressor)
    
    cv = metrics.average_precision_score(y_test, multioutput_regressor.predict(x_test).clip(0.0,1.0))
    
    cvs.append(cv)
    
print(cvs)
print(np.mean(cvs))


# In[15]:


sub = pd.read_csv(path.join(root, 'sample_submission.csv'))
submission = []

for f in test:
    df = pd.read_csv(f)
    df.set_index('Time', drop=True, inplace=True)

    df['Id'] = f.split('/')[-1].split('.')[0]

    dataset = Path(f).parts[-2]
        
#     if dataset == 'tdcsfog':
#         df.AccV = df.AccV / 9.80665
#         df.AccML = df.AccML / 9.80665
#         df.AccAP = df.AccAP / 9.80665
            
    df['Time_frac']=(df.index/df.index.max()).values
    df = pd.merge(df, tasks[['Id','t_group']], how='left', on='Id').fillna(-1)

    df = pd.merge(df, metadata_w_subjects[['Id','Subject', 'Visit','Test','Medication','s_group']], how='left', on='Id').fillna(-1)
    df_feats = fc.calculate(df, return_df=True, include_final_window=True, approve_sparsity=True, window_idx="begin")
    df = df.merge(df_feats, how="left", left_index=True, right_index=True)
    df.fillna(method="ffill", inplace=True)

#     # stride
#     df["Stride"] = df["AccV"] + df["AccML"] + df["AccAP"]

#     # step
#     df["Step"] = np.sqrt(abs(df["Stride"]))
        
    res_vals = []
    
    for i_fold in range(5):
        
        pred = regs[i_fold].predict(df[cols]).clip(0.0,1.0)
        res_vals.append(np.expand_dims(np.round(pred, 3), axis = 2))
        
    res_vals = np.mean(np.concatenate(res_vals, axis = 2), axis = 2)
    res = pd.DataFrame(res_vals, columns=pcols)
    
    df = pd.concat([df,res], axis=1)
    df['Id'] = df['Id'].astype(str) + '_' + df.index.astype(str)
    submission.append(df[scols])
    
submission = pd.concat(submission)
submission = pd.merge(sub[['Id']], submission, how='left', on='Id').fillna(0.0)
submission[scols].to_csv('submission.csv', index=False)


# In[16]:


submission

