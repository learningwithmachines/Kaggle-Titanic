# %%
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from keras.layers import *
import pickle as pkl
import sys, os, csv
from collections import OrderedDict as ODict
from fancyimpute import KNN as impKNN

# %%
# import data
datadir = os.getcwd() + '/data'
datadict, filenames = ODict(), []
for files in os.listdir(datadir):
    if filenames not in filenames:
        filenames.append(files)
    with open(datadir + '/' + files, mode='r') as csvfile:
        datadict[files] = pd.read_csv(csvfile, header=0)
        csvfile.close()
datadict.keys(), filenames

# %%
genderdata = datadict[filenames[-1]]
traindata = datadict[filenames[0]]
testdata = datadict[filenames[1]]

traindata.set_index('PassengerId')
testdata.set_index('PassengerId')

# %%
print(traindata.shape[0], "Rows")
print(genderdata.shape[0], "Rows")
print(testdata.shape[0], "Rows")
# %%
missingtraindata = traindata.isnull().sum()
print(traindata.isnull().sum())

# %%
missingtestdata = testdata.isnull().sum()
print(testdata.isnull().sum())

# %%
# check classes
print(traindata.nunique())

# %%
# set categorical values and create working dataframe copy

catcolumns = ['Pclass', 'Sex', 'Ticket', 'Fare', 'Cabin', 'Embarked']
for _col_ in catcolumns:
    traindata[_col_] = traindata[_col_].astype('category')

traindata['Name'] = traindata.Name.astype(str)

print('All OK!' if (missingtraindata == traindata.isnull().sum()).all() else 'Bad Op')

nomissing = traindata.drop(['Cabin', 'Embarked'], axis=1).copy()
nomissing.isnull().sum()

# %%
# Whats in a name ?

Prefixes = ['Mr.', 'Mrs.', 'Master.', 'Miss.', 'Don.', 'Dr.', 'Rev.', 'Col.', 'Major.', 'Ms.', 'Mme.', 'Lady.', 'Sir.',
            'Mlle.', 'Countess.', 'Capt.', 'Jonkheer.']


def chk_prefix(s) -> str:
    for _pf_ in Prefixes:
        if _pf_ in s:
            return _pf_
    print(s)
    return None


def extract_FamilyName(s) -> str:
    return s.split(',')[0]


Namedata = traindata[['PassengerId', 'Name']].copy()
Namedata['Prefixes'] = traindata.loc[:, ('Name')].astype(str).apply(chk_prefix)
Namedata['FamilyNames'] = traindata.loc[:, ('Name')].astype(str).apply(extract_FamilyName)

print(Namedata.nunique(), "\n\n", Namedata.isnull().sum())

# %%
workingdf = nomissing.drop(['Name', 'Survived'], axis=1).join(Namedata.set_index('PassengerId').drop(['Name'], axis=1),
                                                              on='PassengerId')
print(workingdf.isnull().sum())

# %%
workingdf['Sex'] = workingdf.Sex.astype('category').cat.codes
workingdf['Pclass'] = workingdf.Pclass.astype('category').cat.codes
workingdf['SibSp'] = workingdf.SibSp.astype('category').cat.codes
workingdf['Parch'] = workingdf.Parch.astype('category').cat.codes
workingdf['Ticket'] = workingdf.Ticket.astype('category').cat.codes
workingdf['Fare'] = workingdf.Fare.astype('category').cat.codes
workingdf['Prefixes'] = workingdf.Prefixes.astype('category').cat.codes
workingdf['FamilyNames'] = workingdf.FamilyNames.astype('category').cat.codes
print(workingdf.head())

# %%
from fancyimpute import KNN as impKNN

minage, maxage = workingdf.Age.dropna().min(), workingdf.Age.dropna().max()
newcols = [x for x in workingdf.columns if not x == 'PassengerId']

agepredictor = impKNN(k=5, min_value=minage, max_value=maxage)
ages_df = pd.DataFrame(data=agepredictor.fit_transform(workingdf.drop(['PassengerId'], axis=1)),
                       columns=newcols,
                       index=workingdf.index)

ages_df['PassengerId'] = workingdf['PassengerId']
ages_df['Old_Age'] = workingdf['Age']

# minimum and maximum of precited ages.
print(ages_df[ages_df.Old_Age.isnull()].Age.max(), ages_df[ages_df.Old_Age.isnull()].Age.min())

# %%
# cabin data -  has two parts floor and cabin number.

Cabindata = traindata[["PassengerId", "Cabin"]]


def parse_cabdata(s) -> (str, int):
    ValidFloors = "a b c d e f g".split(" ")
    if type(s) == str:
        if s[0].lower() in ValidFloors and s[1:].isnumeric():
            return s[0], int(s[1:])
    else:
        return np.nan, np.nan


Cabindata[["Floor", "Room"]] = Cabindata.copy().apply(lambda res: pd.Series(parse_cabdata(res["Cabin"])),
                                                      axis=1)
Cabindata.head()

# %%
# Ticket Data

Ticketdata = traindata[["PassengerId", "Ticket"]]


def parse_ticketdata(s) -> (str, int):
    delim = s.find(" ")
    if delim < 0:
        if s.isnumeric():
            return np.nan, np.int(s)
        else:
            return s, np.nan
    else:
        nextlim = s[delim+1:].find(" ")
        print(delim, nextlim)
        if nextlim < 0:
            return str(s[:delim]), np.int(s[delim+1:])
        else:
            delim += nextlim
            return str(s[:delim]), np.int(s[delim+1:])


Ticketdata[["tkt_headr", "tkt_num"]] = Ticketdata.copy().apply(lambda res: pd.Series(parse_ticketdata(res["Ticket"])),
                                                               axis=1)
Ticketdata.head()
