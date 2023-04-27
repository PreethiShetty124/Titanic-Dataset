# Titanic-Dataset

# Objective
The sinking of the Titanic is one of the most infamous shipwrecks in
history. On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t
enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224
passengers and crew. While there was some element of luck involved in surviving, it seems some groups of
people were more likely to survive than others. The objective of this project is to build a predictive model that answers the question: “what sorts of people were more likely to survive?

# Exploratory Data Analysis(EDA)
Applied EDA on all features like Sex, PClass, SIBSP, Age, Parch and Embarked this helped us to gain better insights of data. EDA helps us to analyse the data using all the important features and predict how many people or which class of people survived the most.  

# Feature Engineering
1) Handled Null Values - Train and test data both contained the null values. Handled the null values by using median and mode method
2) Label Encoding - Machine learning is not trained to deal with text data. With the help of label encoder and mapping, converted all text categorical data to numerical data like Sex column,Embarked column, Age and Family.
3) Combining SIBSP and Parch together and creating new Family feature as both belong to family categories
4) Removing unwanted columns like Name, SIBSp, Parch,Ticket and Passenger ID.

# Data Analysis
Applied several machine learning algorithms to get best accuracy like  KNeighbors,Decision Tree Classifier, Random Forest, GaussianNB, Support Vector Machine(SVM) using 3 kernels. Got Best possible accuracy as 83.16 using SVM(Kernel = rbf).  Based on the SVM model, we will predict the results. Results of prediction of how many people 
survived in Titanic is generated in Submission.csv file.
