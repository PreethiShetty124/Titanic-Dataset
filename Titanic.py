# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# loading the data
train = pd.read_csv("/content/train.csv")
train.head()

test = pd.read_csv("/content/test.csv")
test.head()

train.info()
test.info() 

train.isnull().sum() #contains null value
test.isnull().sum() #contains null value

#Visualizing the data
def bar_chart(feature):
    survived = train[train["Survived"]==1][feature].value_counts()
    dead = train[train["Survived"]==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
    
bar_chart('Sex')
#Here we can observes that females are more likely to survive as compared to males.




bar_chart('Pclass')
#Passengers in Pclass1(Upper) are more likely to survive as compared to Pclass2(Middle) and Pclass3(Lower)

bar_chart('SibSp')

plt.hist(train["Age"],edgecolor = "black")
plt.xlabel('Age')
plt.ylabel('count')
plt.show()
#People lying in age group of 20-40 years have more chances of survival as compared to other age group.

bar_chart('Parch')

bar_chart('Embarked')

#Splitting the title from the Name column.
train_test_data = [train, test] # combining train and test dataset
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
train['Title'].value_counts()

test['Title'].value_counts()

# label encoding titles and combining all the less count titles in one category
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    
# after mapping new col title has been created in train
train[:2]

# after mapping new col title has been created in test
test[:2]

bar_chart('Title')
#We can observe here that 0 label(MR) i.e Males has died more.We have drawen the same conclusion when we saw the bar chart using sex col survival rates of males are less as compared to females.

# Handling Null Values and Label Encoding
# Delete Name col from dataset as it is not required.
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
# label encoding sex column
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
# handling null values of Embarked column in train dataset
modeEM = train.Embarked.mode()
modeEM = modeEM[0]
train['Embarked'] = train.Embarked.fillna(modeEM)
# label encoding Embarked Col
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
# handling null values of Fare column in test dataset
test['Fare'].fillna(test['Fare'].median(),inplace = True)
for dataset in train_test_data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
    
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
    
 Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
    
# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

#SIBSP and Parch both belong to family categories hence combining both the column and creating new col as Family
train["Family"] = train["SibSp"] + train["Parch"] + 1
test["Family"] = test["SibSp"] + test["Parch"] + 1

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['Family'] = dataset['Family'].map(family_mapping)
    
  # dropping unnecessary column like SIBSp, Parch,Ticket and Passenger ID
train = train.drop(['SibSp','Parch','Ticket','PassengerId'],axis = 1)
test = test.drop(['SibSp','Parch','Ticket'],axis = 1)

X_train = train.drop('Survived',axis = 1)
Y_train = train['Survived']
print(X_train.shape, Y_train.shape)

#Applying different machine learning models to get the best accuracy
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = KNeighborsClassifier(n_neighbors = 15)
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
print(f"KNN Score : {round(np.mean(score)*100,2)}")

clf1 = DecisionTreeClassifier()
score = cross_val_score(clf1, X_train, Y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
print(f"Decision Tree Score: {round(np.mean(score)*100, 2)}")

clf2 = RandomForestClassifier(n_estimators=200,random_state = 2)
score = cross_val_score(clf2, X_train, Y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
print(f"Random Forest Classifier Score: {round(np.mean(score)*100, 2)}")

clf3 = GaussianNB()
score = cross_val_score(clf3, X_train, Y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
print(f"Gaussian Score: {round(np.mean(score)*100, 2)}")

clf4 = SVC(kernel ='rbf')
score = cross_val_score(clf4, X_train, Y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
print(f"SVC Score: {round(np.mean(score)*100, 2)}")

#We got the best accuracy in SVM(kernel = rbf) hence we are going to predict the model using SVM.
clf4 = SVC(kernel ='rbf')
clf4.fit(X_train, Y_train)
test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf4.predict(test_data)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)

submission = pd.read_csv('submission.csv')
submission.head()

Conclusion:
From the above dataset, we can conclude that features like name, gender, class/berth of the passenger, whether they are travelling alone or with their family members 
are very important to predict what sort of people were able to survive.
 
After performing exploratory data analysis on the Titanic dataset, we can conclude that women were given higher priority as compared to men at the time of evacuating
the ship hence women survived more as compared to men. Also the preference were given to certain group of passengers like passengers travelling with their family 
members, age of the passengers like children, toddlers, young kids were given first preference. Thus we can conclude by saying that certain features or group of 
people were given more preference than others hence these group of people were able to survive.
