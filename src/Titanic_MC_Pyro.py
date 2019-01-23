#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass

#%%
from IPython.core.interactiveshell import InteractiveShell
import os
import sys
import time
import pickle
import multiprocessing
import pixiedust
import PIL as pil
from matplotlib import pyplot as plt
import seaborn as sns
from collections import OrderedDict as ODict
import numpy as np
import pandas as pd

# SKLEARN
from collections import Counter, OrderedDict
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier,VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import GridSearchCV

# KNN-Impute
import tensorflow as tf
from fancyimpute import KNN

# PYTORCH
import torch as tc
from torchvision import transforms, datasets
from torchvision import models as zoo
from torch import nn, optim
from torch.nn import functional as fu

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


InteractiveShell.ast_node_interactivity = 'all'

# %pixie_debugger
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
sns.set(style='white', context='notebook', palette='tab10')
get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')
get_ipython().run_line_magic('config', 'IPCompleter.use_jedi = True')

#%% [markdown]
# ## Import and Check data

#%%
datadir = os.getcwd() + '/../data'
filenames = ['gender_submission.csv', 'train.csv', 'test.csv']
datadict  = ODict()
for files in filenames:
    try:
        with open(datadir + '/' + files, mode='r') as csvfile:
            datadict[files] = pd.read_csv(csvfile, header=0, encoding='ascii')
            csvfile.close()
        print('found file: {}'.format(files))
    except FileNotFoundError:
        print('skipping file ./{}'.format(files))

datadict.keys(), filenames


#%%
# %%
# Print out data, for quick look.

genderdata = datadict[filenames[0]]
traindata = datadict[filenames[1]]
testdata = datadict[filenames[-1]]

print(traindata.shape[0],"Rows")
traindata.info()


#%%
testdata.set_index('PassengerId')
testdata.info()


#%%
genderdata.info()


#%%
# check sample sizes per class per column
len(traindata)/traindata.nunique()

#%% [markdown]
# ## Exploratory data analysis

#%%
train_nulls = (traindata.isnull().sum() / len(traindata)) * 100
train_nulls = train_nulls.drop(train_nulls[train_nulls == 0].index)
test_nulls = (testdata.isnull().sum() / len(testdata)) * 100
test_nulls = test_nulls.drop(test_nulls[test_nulls == 0].index)
train_missing = pd.DataFrame({'Training NaNs' :train_nulls})
train_missing.index.name = 'Metric'
test_missing = (pd.DataFrame({'Test NaNs' :test_nulls}))
test_missing.index.name = 'Metric'
all_missing = pd.merge(train_missing, test_missing, on='Metric', how='outer')
all_missing.head()


#%%
# Fill empty's with NaNs
traindata = traindata.fillna(np.nan)
testdata = testdata.fillna(np.nan)

#%% [markdown]
# ### Exploring our Variables
# 
#     * PassengerId -> Ignore
#     * Pclass -> Categorical -> Survival Probability
#     * Name -> Categorical with Title (after Extraction) -> Survival Probability
#     * Sex -> Categorical -> Survival Probability
#     * Age -> Continous / Categorical (Binned) -> Survival Probability
#     * SibSP -> Categorical -> Survival Probability
#     * Parch -> Categorical -> Survival Probability
#     * Ticket -> ? -> Needs Features
#     * Fare -> Continous / Categorical (Binned) -> Survival Probability
#     * Cabin -> ? (Deck | Deck Position | ?) -> Needs Features
#     * Embarked -> Categorical -> Survival Probability

#%%
# Pclass
ax_pc = sns.countplot(x='Pclass',
                   hue="Survived",
                   data=traindata)

traindata[['Pclass',
           'Survived']].groupby('Pclass').count().sort_values(by='Survived',
                                                                    ascending=False)


#%%
ax_pp = sns.barplot(x='Pclass',
                   y='Survived',
                   data=traindata)

ax_pp.set_ylabel("Survival Probability")

traindata[['Pclass',
           'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived',
                                                             ascending=False)


#%%
# Names

Prefixes = ['Mr.','Mrs.',
            'Master.','Miss.',
            'Don.','Dr.','Rev.',
            'Col.','Major.',
            'Ms.','Mme.','Lady.',
            'Sir.','Mlle.','Countess.',
            'Capt.','Jonkheer.']

def apply_prefix(s) -> str:
    
    Prefixes = ['Mr.','Mrs.','Master.','Miss.',
                'Don.','Dr.','Rev.','Col.',
                'Major.','Ms.','Mme.','Lady.',
                'Sir.','Mlle.','Countess.','Capt.','Jonkheer.']
    
    for _pf_ in Prefixes:
        if _pf_ in s:
            return _pf_
    return None

Pf_data = traindata.loc[:,('Name','Survived')]
Pf_data['Prefix'] = traindata.copy().loc[:,'Name'].astype(str).apply(apply_prefix) 


#%%
P = Pf_data[['Prefix',
             'Survived']].groupby('Prefix').mean().sort_values(by='Survived',
                                                               ascending=False) * 100
C = Pf_data[['Prefix',
             'Survived']].groupby('Prefix').count().sort_values(by='Survived',
                                                                ascending=False)
byNames = pd.merge(P,C,how='outer',on='Prefix')
byNames.columns = ['Survived(%)','Survived(N)']
byNames.index.name='Prefix'

fig = plt.figure(figsize=(12,8))

ax_np = sns.barplot(x=byNames.index,y=byNames['Survived(%)'])
ax_np.set_ylabel('Survival Probability by Title')


#%%
# by Gender

ax_gn = sns.barplot(x='Sex',y='Survived',data=traindata)
ax_gn.set_ylabel('Survival Probability')

traindata[['Sex','Survived']].groupby(['Sex']).mean()


#%%
# by Age - Continuous KDE

fig = plt.figure(figsize=(10,8))

ax = sns.kdeplot(traindata.loc[(traindata['Survived']==1),('Age')],
                 color='b',
                 shade=True,
                 label='Survived')

ax = sns.kdeplot(traindata.loc[(traindata['Survived']==0),('Age')],
                 color='r',
                 shade=True,
                 label='Not Survived')

plt.title('Survivors v/s Non Survivors by Age')
plt.xlabel('Age')
plt.ylabel('Probability')


#%%
# by Age - Discrete Age Binning

fig = plt.figure(figsize=(10,8))

ageDF = traindata[['Age','Survived']]

maxage, minage = int(ageDF.Age.max()), int(ageDF.Age.min())
agebins = np.arange(minage, maxage+10, 10)
agelabels = [str(agebins[i])+'-'+str(agebins[i+1])
             for i in range(len(agebins)-1)]


ageDF['Age_Binned'] = pd.cut(ageDF.loc[:,('Age')],
                             bins=agebins,
                             labels=agelabels)


Binned_Probs = ageDF[['Age_Binned','Survived']].groupby(['Age_Binned']).mean()

ax_ag = sns.barplot(x=Binned_Probs.index,
                    y='Survived', 
                    data=Binned_Probs)

ax_lmag = sns.lmplot('Age','Survived',data=ageDF)

ax_ag.set_ylabel('Survival Probability by Age Group')

Binned_Probs


#%%
# by siblings on board

fig = plt.figure(figsize=(10,8))

sib_data = traindata[['SibSp', 'Survived']].groupby(['SibSp']).mean()

ax_sb = sns.barplot(x=sib_data.index, y='Survived', data=sib_data)
ax_sb.set_ylabel('Survival Probabibility')
ax_sb.set_xlabel('Number of siblings on board')


#%%
# by parents on board

fig = plt.figure(figsize=(10,8))

Parch_data = traindata[['Parch', 'Survived']].groupby(['Parch']).mean()

ax_sb = sns.barplot(x=Parch_data.index, y='Survived', data=Parch_data)
ax_sb.set_ylabel('Survival Probabibility')

Parch_data


#%%
# by fares

fig = plt.figure(figsize=(10,8))

ax = sns.distplot(traindata.loc[(traindata['Survived']==1),('Fare')],
                 color='b',
                 label='Survived')

ax = sns.distplot(traindata.loc[(traindata['Survived']==0),('Fare')],
                 color='r',
                 label='Not Survived')

plt.title('Survivors v/s Non Survivors by Fare')
plt.legend()
plt.xlabel('Fare')
plt.ylabel('Probability')


#%%
# Cabin
traindata['Cabin'].unique()


#%%
# Embarked
ax = sns.countplot(x='Embarked',hue='Survived',data=traindata)
ax.set_ylabel('Count')

traindata[['Embarked','Survived']].groupby(['Embarked']).count()


#%%
# Multiplots
# Age and Pclass

sns.lmplot('Age', 'Survived', data=traindata, hue='Pclass')


#%%
# Age, Sex and Pclass
fig = plt.figure(figsize=(12,4), dpi=96)
sns.catplot(x='Sex', y='Survived', col='Pclass', 
            data=traindata, kind='bar', aspect=0.6)


#%%
# Age, Sex, and Siblings on board
fig = plt.figure(figsize=(28,4), dpi=96)
sns.catplot(x='Sex', y='Survived', col='SibSp', 
            data=traindata, kind='bar', aspect=0.6)


#%%
#Age, Sex and Parch
fig = plt.figure(figsize=(28,4), dpi=96)
sns.catplot(x='Sex', y='Survived', col='Parch', 
            data=traindata, kind='bar', aspect=0.6)

#%% [markdown]
# ## Feature Engineering

#%%
# combining data

traindata['origin'] = 'train'
testdata['origin'] = 'test'
alldata = pd.concat([traindata, testdata], ignore_index=True, sort=False)
print(traindata.shape, testdata.shape, alldata.shape)
print(set(traindata.columns) - set(testdata.columns))
alldata.isnull().sum()


#%%
# passengerID -> droppping

alldata.drop(labels = ['PassengerId'], axis=1, inplace=True)


#%%
# Pclass - dummies
alldata['Pclass'] = alldata.Pclass.astype('category')
alldata = pd.get_dummies(alldata, columns=['Pclass'], prefix='Pclass')


#%%
# Name - Extract Prefixes
alldata['Prefix'] = alldata.astype(str).apply(lambda s: pd.Series(apply_prefix(s["Name"])),axis=1)
print(alldata.Prefix.unique())
alldata = pd.get_dummies(alldata, columns = ['Prefix'], prefix='Has_Title')


#%%
alldata.columns


#%%
#Name -> Drop Raw Names
alldata.drop(labels = ['Name'], axis=1, inplace=True)
alldata.head()


#%%
#Sex -> Categorize

alldata = pd.get_dummies(alldata, columns=["Sex"], prefix="Is")
alldata.head()


#%%
def KNNfill(df,usecols,predcol, knn_k=5):
    
    dfcpy = df.copy().fillna(value=float('NaN')).loc[:, usecols]
    minval = dfcpy.loc[dfcpy[predcol].notnull(),(predcol)].min()
    meanval = dfcpy.loc[dfcpy[predcol].notnull(),(predcol)].mean()
    maxval = dfcpy.loc[dfcpy[predcol].notnull(),(predcol)].max()
    
    predictor = KNN(k=knn_k, min_value=minval, max_value=maxval)
    print("Starting Imputation, Printing NaNs for Passed DataFrame::\n{}\n".format(dfcpy.isnull().sum()))
    print("{} values missing for {}".format(dfcpy[predcol].isnull().sum(),predcol))
    imputed_df = pd.DataFrame(data=predictor.fit_transform(dfcpy),
                             columns=usecols)
    imputed_df['orig_'+predcol] = dfcpy.loc[:, (predcol)]
    return imputed_df


#%%
usecols = ['Is_male','Is_female','Pclass', 'Parch', 'SibSp', 'Age',
           'Has_Title_Capt.','Has_Title_Col.', 'Has_Title_Countess.',
           'Has_Title_Don.','Has_Title_Dr.', 'Has_Title_Jonkheer.',
           'Has_Title_Lady.','Has_Title_Major.', 'Has_Title_Master.',
           'Has_Title_Miss.','Has_Title_Mlle.', 'Has_Title_Mme.',
           'Has_Title_Mr.', 'Has_Title_Mrs.','Has_Title_Ms.',
           'Has_Title_Rev.', 'Has_Title_Sir.']
imputed_df = KNNfill(alldata, usecols, 'Age')
alldata['Age'] = imputed_df.Age

print("OK!" if alldata.Age.isnull().sum()==0 else "Imputation Failed!")

imputed_df.loc[imputed_df['orig_Age'].isnull(),('orig_Age','Age')].head()


#%%
# discretize age to 5 year bins.

alldata['Age'] = alldata['Age'].map(lambda s: int(int(s)//5))
alldata['Age'] = alldata['Age'].astype(np.int8)
print(sorted(list(alldata['Age'].unique())))


#%%
fig = plt.figure(figsize=(10,8))

ax_ag0 = fig.add_axes([0.0, 0.5, 1.0, 0.4]) # lbwh
ax_ag1 = fig.add_axes([0.0, 0.0, 1.0, 0.4])

ax_ag0 = sns.distplot(alldata.loc[(alldata['Survived']==1),('Age')],
                 color='b',
                 label='Survived', ax=ax_ag0)

ax_ag_1 = sns.distplot(alldata.loc[(alldata['Survived']==0),('Age')],
                 color='r',
                 label='Not Survived', ax=ax_ag1)

plt.title('Survivors v/s Non Survivors by Grouped Age',
          loc='center', pad = 300 )
ax_ag0.legend()
ax_ag0.set_ylabel('Probability')
ax_ag0.set_xlabel('Age Group')
ax_ag0.set_xlim(-1,17)
ax_ag1.legend()
ax_ag1.set_ylabel('Probability')
ax_ag1.set_xlabel('Age Group')
ax_ag1.set_xlim(-1,17)


#%%
# dummify


alldata = pd.get_dummies(alldata, columns=['Age'], prefix='In_AgeGRP')


#%%
# Age - Use Sex, Pclass, Parch, SibSp, Prefix to Fill Age

from fancyimpute import KNN

minage = alldata.loc[alldata['Age'].notnull(),('Age')].min()
maxage = alldata.loc[alldata['Age'].notnull(),('Age')].max()
medianage = alldata.loc[alldata['Age'].notnull(),('Age')].mean()

cols = ['Sex', 'Pclass', 'Parch', 'SibSp', 'Prefix', 'Age']
targetdf = alldata.fillna(value=float('NaN')).copy().loc[:, cols]
predictage = KNN(k=5, min_value=minage, max_value=maxage)
imp_agesdf = pd.DataFrame(data=predictage.fit_transform(targetdf),
                          columns=cols)

imp_agesdf['orig_ages'] = targetdf.loc[:, ('Age')]
imp_agesdf.loc[imp_agesdf['orig_ages'].isnull() == True].head()

imp_minage = imp_agesdf.loc[alldata['Age'].notnull(),('Age')].min()
imp_maxage = imp_agesdf.loc[alldata['Age'].notnull(),('Age')].max()
imp_medianage = imp_agesdf.loc[alldata['Age'].notnull(),('Age')].mean()

print("Min:{}->{},Mean:{}->{},Max:{}->{}".format(minage,imp_minage,
                                                medianage, imp_medianage,
                                                maxage, imp_maxage))


#%%
# SibSp and Parch -> Family Size

# switch of scipy.stats warning for enforced use of arr[np.array(seq)]
np.warnings.filterwarnings('ignore')

alldata[('FamilySize')] = alldata[('SibSp')] + alldata[('Parch')] + 1

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)

ax = sns.barplot(x='FamilySize',y='Survived',data=alldata,alpha=0.5)
for c in ax.patches:
    c.set_zorder(0)

ax2 = ax.twinx()
sns.lineplot(x=np.arange(0, len(alldata.FamilySize.unique())),
                        y=alldata.FamilySize.value_counts(),
             ax=ax2)

ax2.set_ylabel('Frequency')


#%%
# create newer features for family sizes
alldata['FS_Single'] = alldata['FamilySize'].map(lambda s: 1 if s == 1 else 0)
alldata['FS_Small'] = alldata['FamilySize'].map(lambda s: 1 if s == 2 else 0)
alldata['FS_Medium'] = alldata['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
alldata['FS_Large'] = alldata['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


#%%
# Cat Codes for SibSp and Parch

alldata = pd.get_dummies(alldata, columns=["SibSp"])
alldata = pd.get_dummies(alldata, columns=["Parch"])

alldata.columns


#%%
def parse_ticket(s:str)->(str, int):
    if s.strip(" ").isnumeric():
        return float('NaN'), int(s)
    elif s.strip(" ").isalpha():
        return s.strip(" "), int(0)
    else:
        try:
            xstr = s.replace('.','').replace('/','').strip(' ').split(' ')
            s,i = ''.join(xstr[0:-1]),xstr[-1]
            return s, int(i)
        except:
            xstr = s.replace('.','').replace('/','').strip(' ').split(' ')
            ss,i = ''.join(xstr[0:-1]),xstr[-1]
            print(s,ss, i)
    
alldata[["TKTHEADER","TKTNUM"]] = alldata.apply(lambda s: pd.Series(parse_ticket(s["Ticket"])),
                                                               axis=1)
alldata[['TKTHEADER','TKTNUM']].head()


#%%
# cleaning up
alldata['TKTNUM'] = alldata.TKTNUM.astype(np.int32)
alldata['TKTHEADER'] = alldata.TKTHEADER.astype(str)
alldata.drop(labels=['Ticket'], axis=1, inplace=True)

alldata = pd.get_dummies(alldata, columns = ['TKTHEADER'], prefix='TKT_HEADER')

list(alldata.columns)


#%%
# Fares
print(alldata['Fare'].isnull().sum())

usecols = set(alldata.columns) - set(['Survived','Cabin',
                                      'origin','Embarked',])
predcol = 'Fare'

imputed_df = KNNfill(alldata,usecols,predcol,100)


#%%
comp_fares = (float(imputed_df.loc[imputed_df['orig_Fare'].isnull(),('Fare')]),
              imputed_df.orig_Fare.mean())

print('Median:{}, Predicted:{}'.format(*comp_fares))

# logshift fares value 
alldata['Fare'] = np.log1p(imputed_df['Fare'])

sns.distplot(alldata['Fare'])


#%%
# discretizing fares

alldata['Fare'] = alldata['Fare'].map(lambda s: int(s/0.5))
alldata = pd.get_dummies(alldata, columns=['Fare'], prefix='Fare_group')


#%%
print(list(alldata.columns))


#%%
# Cabins

def parse_cabin(s):
    
    Decks = ['A','B','C','D','E','F','G','T']
    DeckSet = set()
    sc = s
    
    if type(s)==float:
        return float('NaN'), float('NaN')
    else:
        for deck_ in Decks:
            if deck_ in s:
                DeckSet.add(deck_)
                s = s.replace(deck_,"")
        if s == "":
            return tuple(DeckSet), float('NaN')
        else:
            Rooms = tuple(int(_s) for _s in s.strip(' ').split(' '))
            return tuple(DeckSet), tuple(Rooms)
        
alldata[['Deck','Cabin']] = alldata.apply(lambda s: pd.Series(parse_cabin(s["Cabin"])),axis=1)
alldata.loc[alldata['Deck'].notnull(), ('Deck','Cabin')].head()


#%%
alldata['Deck'] = alldata['Deck'].map(lambda s: 'N/A' if type(s)!=tuple else ''.join(s))
alldata['Cabin'] = alldata['Cabin'].map(lambda s: 'N/A' if type(s)!=tuple else len(s))
alldata.Deck.unique(), alldata.Cabin.unique()


#%%
ax_dk  = sns.factorplot(x="Deck",y="Survived",data=alldata,kind="bar"
                        ,order=['A','B','C','D','E','F','FE','G','GF','T','N/A'])
ax_dk.despine(left=True)
ax_dk = ax_dk.set_ylabels("Survival Probabilities V/s Decks")


#%%
ax_cb  = sns.factorplot(x="Cabin",y="Survived",data=alldata,kind="bar"
                        ,order=[1,2,3,4,'N/A'])
ax_cb.despine(left=True)
ax_cb = ax_cb.set_ylabels("Survival Probabilities V/s Allocated Rooms")


#%%
alldata = pd.get_dummies(alldata, columns=['Deck'], prefix='Deck')
alldata = pd.get_dummies(alldata, columns=['Cabin'], prefix='Num_Rooms_is')
alldata.columns


#%%
# Embarked
alldata['Embarked'].unique()


#%%
EmbMap = {'C':0, 'Q':1, 'S':2}

alldata['Embarked'] = alldata['Embarked'].map(lambda s: np.nan if type(s)==float else EmbMap[str(s)])

alldata['Embarked'].isnull().sum()


#%%
usecols = set(alldata.columns) - set(['Survived','origin'])

predcol = 'Embarked'

imputed_df = KNNfill(alldata,usecols,predcol,100)


#%%
alldata['Embarked'] = imputed_df['Embarked']
alldata['Embarked'] = alldata.Embarked.map(lambda s: int(s))

alldata.Embarked.unique()


#%%
#dummies
alldata = pd.get_dummies(alldata, columns=['Embarked'], prefix='Embarked_at')


#%%
tktnumbers = alldata.TKTNUM
alldata.drop(labels=['TKTNUM'], axis=1, inplace=True)


#%%
# forced typecasting

uint8check = lambda x: True if x.max() <= 255 else False

for col in alldata.columns:
    safestr = ''
    try:
        safestr = 'YES' if uint8check(alldata[col]) else '(!) NO'
        if uint8check(alldata[col]):
            alldata[col] = alldata[col].astype(np.uint8)
    except:
        print('(!) Exception at: ',col)
        safestr = '(!) NO'
    print("{}: UINT8 Safe:\033[1m{}\033[0m, Type:{}".format(col,
                                            safestr,
                                            alldata[col].dtype))


#%% [markdown]
# # Model Building

#%%
# prepare testing and training datasets

train = alldata.copy().loc[alldata['origin']=='train']
test = alldata.copy().loc[alldata['origin']=='test']


train.drop(labels=['origin'], axis=1, inplace=True)
train['Survived'] = train['Survived'].astype(np.uint8)
test.drop(labels=['Survived', 'origin'], axis = 1, inplace=True)

train.shape, test.shape


#%%
X_train, Y_train = train.drop(labels=['Survived'], axis=1), train.Survived.astype(np.int8)

X_train.shape, Y_train.shape, type(Y_train)


#%%
# cobbling together our algorithms

random_seed = 42
nCPU = 8
n_splits=9

# preparing for cv-folds
kfold = StratifiedKFold(n_splits)

classifiers = OrderedDict()
classifiers['SVC'] = SVC(random_state=random_seed)
classifiers['DecisionTree'] = DecisionTreeClassifier(random_state=random_seed)
classifiers['MLPClassifier'] = MLPClassifier(random_state=random_seed)
classifiers['KNN'] = KNeighborsClassifier()
classifiers['LinDA'] = LinearDiscriminantAnalysis()
classifiers['LogRegress'] = LogisticRegression(random_state=random_seed)
classifiers['RandomForest'] = RandomForestClassifier(random_state=random_seed)
classifiers['ADAboost'] = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=random_seed),
                                            random_state=random_seed, learning_rate=1e-3)
classifiers['Gradboost'] = GradientBoostingClassifier(random_state=random_seed)
classifiers['ExtraTrees'] = ExtraTreesClassifier(random_state=random_seed)

# Voting is for the last

classifiers


#%%
cv_scores = OrderedDict()

if __name__ == '__main__':
    for _key_ in classifiers.keys():
        cv_scores[_key_] = cross_val_score(classifiers[_key_],
                                            X_train, y=Y_train,
                                            scoring='accuracy',
                                            cv=kfold, n_jobs=nCPU,
                                            verbose=True)

cv_means, cv_std = OrderedDict(), OrderedDict()
for _key_ in classifiers.keys():
    cv_means[_key_] = cv_scores[_key_].mean()
    cv_std[_key_] = cv_scores[_key_].std()


cv_results = pd.DataFrame(data=list(classifiers.keys()), columns=['Algorithms'])
for _ix_ in range(n_splits):
    for _k_ in classifiers.keys():
        cv_results.loc[cv_results['Algorithms']==_k_,
                       'score_k='+str(_ix_)] = cv_scores[_k_][_ix_]
        cv_results['mean_score'] = cv_results['Algorithms'].map(cv_means)
        cv_results['stdev_score'] = cv_results['Algorithms'].map(cv_std)

cv_results[['Algorithms', 'mean_score']]


#%%
fig = plt.figure(figsize=(10,8))
ax_cvres = sns.barplot('mean_score', 'Algorithms', data=cv_results,
                       orient='h', xerr=cv_results.stdev_score)
ax_cvres.set_xlabel('Mean Accuracies')
axes = ax_cvres.axes
axes.set_xlim(0,1.0)
axes.set_xticks(np.arange(0,1.0, 0.025), minor=True)
axes.grid(color='gray', which='both', axis='x', alpha=0.5)
ax_cvres.set_title('Cross Validation Scores')

#%% [markdown]
# #### Tuning with GridSearchCV

#%%
GS_classifiers = OrderedDict()

def do_mpfit(model, grid_params, X=X_train, y=Y_train, strat_cv=kfold):
    
    gsearch_model = GridSearchCV(model, param_grid=grid_params,
                                 cv=strat_cv, scoring='accuracy',
                                 n_jobs=nCPU, pre_dispatch=2*nCPU,
                                 verbose=True)
    gsearch_model.fit(X, y)
    
    return gsearch_model.best_score_, gsearch_model.best_estimator_


#%%
#SVC
GS_SVC = SVC()

#gridparams
svc_grid_params = {'kernel' : ['rbf'],
                  'gamma' : [0.001, 0.01, 0.1],
                  'C' : [1, 10],
                  'degree' : [2, 3],
                  'shrinking' : [True],
                  'probability' : [True],
                  'decision_function_shape' : ['ovr'],
                  'tol' : [0.001],
                  'random_state' : [random_seed]}

if __name__ == '__main__':
    GS_classifiers['SVC'] = do_mpfit(GS_SVC, svc_grid_params)

print(GS_classifiers['SVC'][:]) #83.27


#%%
#RandomForest

GS_RFC = RandomForestClassifier()

#gridparams

rfc_grid_params = {'n_estimators' : [100, 300],
                  'max_features' : [1, 3, 10],
                  'max_depth' : [7, 9, 11],
                  'criterion' : ['gini', 'entropy'],
                  'min_samples_split' : [2],
                  'min_samples_leaf' : [1],
                  'bootstrap' : [False],
                  'random_state' : [random_seed]
                  }

if __name__ == '__main__':
    GS_classifiers['RFC'] = do_mpfit(GS_RFC, rfc_grid_params)

print(GS_classifiers['RFC'][:]) #83.05


#%%
#ADABoost

AdaDTC = DecisionTreeClassifier()

GS_ABC = AdaBoostClassifier(AdaDTC, random_state=random_seed)

#gridparams

abc_grid_params = {'base_estimator__criterion' : ['gini', 'entropy'],
                  'base_estimator__splitter' : ['best', 'random'],
                  'base_estimator__max_depth' : [3, 5],
                  'base_estimator__max_features' : ['sqrt'],
                  'base_estimator__min_samples_split' : [2],
                  'base_estimator__min_samples_leaf' : [1],
                  'base_estimator__random_state' : [random_seed],
                  'base_estimator__presort' : [False],
                  'algorithm' : ['SAMME', 'SAMME.R'],
                  'n_estimators' : [50, 100],
                  'learning_rate' : [0.01, 0.001]              
                  }

if __name__ == '__main__':
    GS_classifiers['ABC'] = do_mpfit(GS_ABC, abc_grid_params)
    
print(GS_classifiers['ABC'][:]) #83.95


#%%
#GradientBoost

GS_GBC = GradientBoostingClassifier()

#gridparams

gbc_grid_params = {'learning_rate' : [0.01, 0.05],
                  'n_estimators' : [200, 300],
                  'max_depth' : [3, 4, 5],
                  'min_samples_split' : [0.1, 0.2, 0.3],
                  'min_samples_leaf' : [1],
                  'max_features' : ['log2'],
                  'validation_fraction' : [0.0],
                  'random_state' : [random_seed]
                  }

if __name__ == '__main__':
    GS_classifiers['GBC'] = do_mpfit(GS_GBC, gbc_grid_params)

print(GS_classifiers['GBC'][:]) #84.06


#%%
#ExtraTrees

GS_ETC = ExtraTreesClassifier()

#gridparams

etc_grid_params = {'n_estimators' : [50, 75, 100],
                  'criterion' : ['gini'],
                  'max_depth' : [4, 5, 6],
                  'min_samples_split' : [2, 0.05],
                  'min_samples_leaf' : [1],
                  'max_features' : ['sqrt'],
                  'random_state' : [random_seed]
                  }

if __name__ == '__main__':
    GS_classifiers['ETC'] = do_mpfit(GS_ETC, etc_grid_params)

print(GS_classifiers['ETC'][:]) #82.82


#%%
# saving model details

with open('../data/exported/GS_classifiers.dat', 'wb') as file:
    pickle.dump(GS_classifiers, file, protocol = -1)
    file.close()


#%%
# load saved file:

with open('../data/exported/GS_classifiers.dat', 'rb') as file:
    GSC_file = pickle.load(file, errors='strict')
    file.close()
    
GSC_file.keys()


#%%
X_train.shape, Y_train.shape

#%% [markdown]
# ### MLP

#%%
class mlp_classifier(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_layers,
                 drop_p):
        
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size,
                                                      hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1],
                          hidden_layers[1:])
        
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)

        
    def forward(self, x):
        
        for each in self.hidden_layers:
            x = fu.relu(each(x))
        x = self.output(x)
        
        return x
    
class customdatasetfromPD(tc.utils.data.Dataset):
    
    def __init__(self, pd, label_col='Survived'):
        
        self.data = pd
        self.length = len(self.data)
        # self.width = self.data.shape[-1] - 1
        self.inp = np.asarray(self.data.drop(labels=[label_col], axis=1))
        self.inp = tc.from_numpy(self.inp.astype(np.float32))
        self.labels = np.asarray(self.data.loc[:,[label_col]])
        self.labels = tc.from_numpy(self.labels.astype(np.float32))
        
    def __getitem__(self, index):
        
        single_input = self.inp[index]
        single_label = self.labels[index]
        
        return single_input, single_label
    
    def __len__(self):
        
        return self.length


#%%
mlp_train = train.sample(frac=0.8, random_state=42)
mlp_test = train.drop(mlp_train.index)

mlp_trainset = customdatasetfromPD(mlp_train)
mlp_testset = customdatasetfromPD(mlp_test)

mlp_trainloader = tc.utils.data.DataLoader(dataset=mlp_trainset,
                                           batch_size=128,
                                           shuffle=True,
                                           pin_memory=True)

mlp_testloader = tc.utils.data.DataLoader(dataset=mlp_testset,
                                          batch_size=128,
                                          shuffle=True,
                                          pin_memory=True)

#%%
class mlp_trainer:

    def __init__(self, model, trainloader, testloader,
                 optimizer, loss_fn, metrics):
        
        self.model = model

        self.trainloader = trainloader
        self.testloader = testloader

        self.lossfn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.trn_records = {k:[] for k in self.metrics.keys()}
        self.val_records = {k:[] for k in self.metrics.keys()}
        self.bwise_losses = []
        
        self.run = self.__call__
    
    def __call__(self, max_epochs):
        
        trainer = create_supervised_trainer(self.model,
                                            self.optimizer,
                                            self.lossfn)

        evaluator = create_supervised_evaluator(self.model,
                                                self.metrics)
        
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_losses(trainer):
            
            records = trainer.state.output 
            self.bwise_losses.append(records)
            
            print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch,
                                                  trainer.state.output))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(self.trainloader)
            metrics = evaluator.state.metrics
            records = list(metrics[k] for k in self.metrics.keys())
            for k in self.metrics.keys():
                self.trn_records[k].append(metrics[k]) 
            
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f}            Avg loss: {:.2f}".format(trainer.state.epoch,*records))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(self.testloader)
            metrics = evaluator.state.metrics
            records = list(metrics[k] for k in self.metrics.keys())
            for k in self.metrics.keys():
                self.val_records[k].append(metrics[k]) 
            
            print("Validation Results - Epoch: {}  Avg accuracy: {:.2f}            Avg loss: {:.2f}".format(trainer.state.epoch, *records))
        
        trainer.run(self.trainloader, max_epochs)


#%%
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from functools import partial

pyro.set_rng_seed(101)

#%%
test_model = mlp_classifier(127,1, [127, 63, 31, 15, 7, 3], 0.25)
log_sigmoid = nn.LogSigmoid()
log_sigmoid

#%%
def pyromodelfn(x_data, y_data, model=test_model):

    priors = {}
    for idx, layer in enumerate(model.hidden_layers):
        name = lambda s,x: f'{s}.{idx}.{x}'
        priors[name('hidden',
               'weights')] = partial(Normal,
                                    loc = tc.zeros_like(layer.weight),
                                    scale = tc.ones_like(layer.weight))
        priors[name('hidden',
               'bias')] = partial(Normal,
                                    loc = tc.zeros_like(layer.bias),
                                    scale = tc.ones_like(layer.bias))

    priors[name('output',
               'weights')] = partial(Normal,
                                    loc = tc.zeros_like(model.output.weight),
                                    scale = tc.ones_like(model.output.weight))

    priors[name('output',
               'weights')] = partial(Normal,
                                    loc = tc.zeros_like(model.output.bias),
                                    scale = tc.ones_like(model.output.bias))

    lifted_module = pyro.random_module('module', model, priors)
    lifted_reg_model = lifted_module()

    lhat = log_sigmoid(lifted_reg_model(x_data))
    pyro.sample('obs', Categorical(logits=lhat),obs=y_data)

def pyroguidefn(x_data, y_data, model=test_model):

    priors, params = {}, {}
    for idx, layer in enumerate(model.hidden_layers):
        name = lambda s,x: f'{s}.{idx}.{x}'
        params[name('hidden','w_mu')] = partial(tc.randn_like, layer.weight)
        params[name('hidden','w_sigma')] = partial(tc.randn_like, layer.weight)
        params[name('hidden','w_mu_param')] = partial(pyro.param,
                                                     name('hidden','w_mu'),
                                                     params[name('hidden','w_mu')])
        params[name('hidden','w_sigma_param')] = partial(pyro.param,
                                                        name('hidden','w_sigma'),
                                                        params[name('hidden','w_sigma')])

        priors[name('hidden','w_prior')] = partial(Normal, loc = params[name('hidden','w_mu_param')],
                                                   scale = params[name('hidden','w_sigma_param')])

        params[name('hidden','b_mu')] = partial(tc.randn_like, layer.bias)
        params[name('hidden','b_sigma')] = partial(tc.randn_like, layer.bias)
        params[name('hidden','b_mu_param')] = partial(pyro.param,
                                                      name('hidden','b_mu'),
                                                      params[name('hidden','b_mu')])
        params[name('hidden','b_sigma_param')] = partial(pyro.param,
                                                         name('hidden','b_sigma'),
                                                         params[name('hidden','b_sigma')])

        priors[name('hidden','b_prior')] = partial(Normal, loc = params[name('hidden','b_mu_param')],
                                                   scale = params[name('hidden','b_sigma_param')])

        lifted_module = pyro.random_module('module', model, priors)

        return lifted_module()

#%%
pyromodel = partial(pyromodelfn, model=test_model)
pyroguide = partial(pyroguidefn, model=test_model)
pyro_optim = Adam({'lr':0.001})
svi = SVI(pyromodel, pyroguide, pyro_optim, loss=Trace_ELBO())

#%%
num_iterations = 5
loss = 0

for j in range(num_iterations):
    loss = 0
    for batch_id, data in enumerate(mlp_trainloader):
        # calculate the loss and take a gradient step
        loss += svi.step(data[0], data[-1])
    normalizer_train = len(mlp_trainloader.dataset)
    total_epoch_loss_train = loss / normalizer_train
    
    print("Epoch ", j, " Loss ", total_epoch_loss_train)

#%%
pyromodel(mlp_trainloader)