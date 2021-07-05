import pandas as pd

test_data_set  = pd.read_csv('test.csv')
train_data_set = pd.read_csv('train.csv')
gen_sub_set    = pd.read_csv('gender_submission.csv')


test_set = gen_sub_set.merge(test_data_set,how='left')
Data_Set = pd.concat([train_data_set,test_set],axis=0)
Data_Set['Sex']=Data_Set['Sex'].replace('male',1).replace('female',0).astype('int')
Data_Set['Title'] = Data_Set['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
Data_Set['Title'] = Data_Set["Title"].map(title_mapping)

Data_Set.drop('Name', inplace=True,axis=1)


Data_Set["Age"].fillna(Data_Set.groupby("Title")["Age"].transform("median"), inplace= True)

Data_Set.loc[Data_Set['Age']<=16,'Age']=0
Data_Set.loc[(Data_Set['Age']>16) & (Data_Set['Age']<=26),'Age']=1
Data_Set.loc[(Data_Set['Age']>26) & (Data_Set['Age']<=36),'Age']=2
Data_Set.loc[(Data_Set['Age']>36) & (Data_Set['Age']<=46),'Age']=3
Data_Set.loc[Data_Set['Age']>46,'Age']=4


Data_Set[Data_Set['Pclass']==1]['Embarked'].value_counts()
Data_Set[Data_Set['Pclass']==2]['Embarked'].value_counts()
Data_Set[Data_Set['Pclass']==3]['Embarked'].value_counts()

Data_Set.loc[Data_Set['Embarked'].isnull()==True,'Embarked']='S'

embarked_dict ={'S':0,'C':1,'Q':2}
Data_Set['Embarked']=Data_Set['Embarked'].map(embarked_dict)

import seaborn as sns
import matplotlib.pyplot as plt

facet = sns.FacetGrid(Data_Set, hue="Survived",aspect=4 )
facet.map(sns.kdeplot, 'Fare', shade = True)
facet.set(xlim = (0, Data_Set['Fare'].max()))
facet.add_legend()
plt.show()

facet = sns.FacetGrid(Data_Set, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, Data_Set['Fare'].max()))
facet.add_legend()
plt.xlim(0, 20)


Data_Set.loc[Data_Set['Fare']<=17, 'Fare']=0
Data_Set.loc[(Data_Set['Fare']>17) & (Data_Set['Fare']<=30), 'Fare']=1
Data_Set.loc[(Data_Set['Fare']>30) & (Data_Set['Fare']<=100),'Fare']=2
Data_Set.loc[Data_Set['Fare']>100,'Fare']=3

Data_Set['Cabin']=Data_Set['Cabin'].str[:1]


frame=[]
for i in range(1,4):
	frame.append(Data_Set[Data_Set['Pclass']==i]['Cabin'].value_counts())
df=pd.DataFrame(frame)
df.index=['1st','2nd','3rd']
df.plot(kind='bar',stacked=True,figsize=(10,5))

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
Data_Set['Cabin']=Data_Set['Cabin'].map(cabin_mapping)

Data_Set['Cabin'].fillna(Data_Set.groupby('Pclass')['Cabin'].transform('median'),inplace=True)

Data_Set['FamilySize']=Data_Set['SibSp']+Data_Set['Parch']+1

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
Data_Set['FamilySize']=Data_Set['FamilySize'].map(family_mapping)

for i in ['Ticket','SibSp','Parch']:
	Data_Set.drop(i,axis=1,inplace=True)

Data_Set.reset_index(drop=True,inplace=True)

x=Data_Set.iloc[:,2:]
y=Data_Set.iloc[:,2]

x_train,x_test = x[:891],x[891:]
y_train,y_test = y[:891],y[891:]


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, x_train,y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

import numpy as np 
clf = [KNeighborsClassifier(n_neighbors = 13),DecisionTreeClassifier(),
       RandomForestClassifier(n_estimators=13),GaussianNB(),SVC(),ExtraTreeClassifier(),
      GradientBoostingClassifier(n_estimators=10, learning_rate=1,max_features=3, max_depth =3, random_state = 10),AdaBoostClassifier(),ExtraTreesClassifier()]
def model_fit():
	scoring = 'accuracy'
	for i in range(len(clf)):
		score = cross_val_score(clf[i], x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
		print("Score of Model",i,":",round(np.mean(score)*100,2))
model_fit()

svc=SVC()
svc.fit(x_train,y_train)

predict_value=svc.predict(np.nan_to_num(x_test))

from sklearn.metrics import accuracy_score 

print ('Accuracy Score : ',accuracy_score(y_test,predict_value))

df1=np.arange(892,1310)
submission_csv = pd.DataFrame()
submission_csv['PassengerId']=df1
submission_csv['Survived']=predict_value
submission_csv.to_csv('submission.csv',index=False)