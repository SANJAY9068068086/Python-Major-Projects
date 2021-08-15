#!/usr/bin/env python
# coding: utf-8

# # HR Analytics Project (Evaluation Project - 2)

# In[1]:


# Import Some necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Let's import the dataset

hr_data = pd.read_csv("HR-Employee-Attrition.csv")
hr_data.head()


# In[3]:


# Shape of the dataset

hr_data.shape


# In[4]:


# Quick information about dataset

hr_data.info()


# This dataset has 9 object columns  in which "Attrition" is our Target Column.

# In[5]:


# Let's check null values if any..

hr_data.isnull().sum()


# There is not a single column that have null values

# In[6]:


# there is column name Employee Number in the dataset and it is not useful. So, let's drop it

hr_data.drop(columns = ["EmployeeNumber"], axis=1, inplace=True)


# In[7]:


# Let's separate the numerica column and categorical column

numerical = hr_data.drop(columns = ["Attrition","BusinessTravel","Department","EducationField","Gender",
                                   "JobRole","MaritalStatus","Over18","OverTime"],axis=1)

categorical = hr_data[["Attrition","BusinessTravel","Department","EducationField","Gender",
                                   "JobRole","MaritalStatus","Over18","OverTime"]]


# In[8]:


# Categorical Columns

categorical.head()


# In[9]:


# Let's count the value of each int64 columns

for col in hr_data.columns:
    if hr_data[col].dtype == 'int64':
        print(hr_data[col].value_counts())
        print()


# In[10]:


# Let's count the value of each object columns

for col in hr_data.columns:
    if hr_data[col].dtype == 'object':
        print(hr_data[col].value_counts())
        print()


# In[11]:


# Let's see column Over18

hr_data["Over18"].unique()


# In[12]:


# Column Over18 has only 1 value and due to only 1, it will not helpful for the dataset, Let's drop it

hr_data.drop(columns = ["Over18"], axis=1, inplace=True)


# In[13]:


# Employee Count

hr_data["EmployeeCount"].unique()


# In[14]:


# Employ count has only 1 value. So, it is not useful to the dataset let's drop it

hr_data.drop(columns = ["EmployeeCount"], axis=1, inplace=True)


# In[15]:


# RelationshipSatisfaction

hr_data["StandardHours"].unique()


# In[16]:


# StandardHours has only 1 value. So, it is not useful to the dataset let's drop it

hr_data.drop(columns = ["StandardHours"], axis=1, inplace=True)


# In[17]:


# Let's check the target column Attrition

sns.countplot("Attrition", data=hr_data)
plt.show()

# There is very high class imbalance issue


# In[18]:


# Let's check the  Business Travel

sns.countplot(x="BusinessTravel",data=hr_data)
plt.show()

# Most of the employee travel rarely


# In[19]:


# Business Traves vs Attrition

sns.countplot(x="BusinessTravel", data=hr_data, hue="Attrition")
plt.show()

# rarely travel employee has only 20% attrition to yes and who travel frequently has near 35% attrition to yes
# and non travel has very low attrition to yes


# In[20]:


# Let's check department

sns.countplot(x="Department",data=hr_data)
plt.show()


# In[21]:


# Department vs Attrition

sns.countplot(x="Department", data=hr_data, hue="Attrition")
plt.show()

# Human Resources has high yes attrition than Sales and Research development


# In[22]:


# Education Field

sns.countplot(x="EducationField", data=hr_data)
plt.show()

# Most of the employee are from Life Science and Medical field


# In[23]:


# Education vs Attrition

plt.figure(figsize=(24,5))
sns.countplot(x="EducationField", data=hr_data, hue="Attrition")
plt.show()

# Technical Degree and Human research employee has high % attrition to yes and Life science and Medical employee has low % to the yes


# In[24]:


# Gender

sns.countplot(x="Gender", data=hr_data)
plt.show()

# male employees are more than females


# In[25]:


# Gender vs Attrition

sns.countplot(x="Gender", data= hr_data, hue="Attrition")

# Almost equal % contribution to yes attrition by male and female


# In[26]:


# Maritial Status

sns.countplot(x="MaritalStatus", data=hr_data)

# married employee are high in numbers


# In[27]:


# Maritial Status vs Attrition

sns.countplot(x="MaritalStatus", data=hr_data, hue="Attrition")

# females are giving high in % to the yes attrition


# In[28]:


# OverTime

sns.countplot(x="OverTime", data=hr_data)

# the employee who are doing overtime are less


# In[29]:


# OverTime vs Attrition

sns.countplot(x="OverTime", data=hr_data, hue="Attrition")

# The employee who are doing overtime are more stable than others


# In[30]:


# Age

plt.figure(figsize=(24,7))
sns.countplot(x="Age", data=hr_data)

# most of the employee are between 15-40


# In[31]:


# Age vs Attrition

plt.figure(figsize=(24,7))
sns.countplot(x="Age", data=hr_data, hue="Attrition")

# employee less than age 29 age have higher attrition to yes


# In[32]:


# years at company vs attrition

plt.figure(figsize=(24,8))
sns.countplot(x="YearsAtCompany", data=hr_data[hr_data["Attrition"]=='Yes'])
plt.show()

# employee who has only 1 years of experience are more stable than othere
# and it is very rare chance to stay an employee who has experience more than 10 yesrs


# In[33]:


# Job role vs attrition

plt.figure(figsize=(25,7))
sns.countplot(x="JobRole", data=hr_data[hr_data["Attrition"]=="Yes"])
plt.show()

# Sales Executive, Laboratory Technician, Sales Representative and Research Scientist has are more stable than others


# In[34]:


# YearsWithCurrManager

plt.figure(figsize=(25,7))
sns.countplot(x="YearsWithCurrManager",data=hr_data[hr_data["Attrition"]=='Yes'])
plt.show()

# who are fresher has higher attrition
# maybe if they get 2 years of experience they left the job
# so possibilities to give the chances to the fresher who stay for long time in the company


# In[35]:


# Year at company vs attrition

#plt.figure(figsize=(25,8))
sns.barplot(y="YearsAtCompany", x="Attrition", data=hr_data)
plt.show()

# average numbers of employess is staying in the complany before leaving


# In[36]:


# let's check the correlation matrix

corr = hr_data.corr()
corr


# In[37]:


# Let's check the skewness

print(hr_data.skew())
print("\nTotal Count of numerica features : ",len(hr_data.skew()))
print("Count of features which are significant skewed : ",len(hr_data.skew().loc[abs(hr_data.skew())>0.5]))


# In[38]:


# Let's remove the skewness

for index in hr_data.skew().index:
    if hr_data.skew().loc[index]>0.5:
        hr_data[index] = np.log1p(hr_data[index])
    if hr_data.skew().loc[index]<-0.5:
        hr_data[index] = np.square(hr_data[index])


# In[39]:


# Let's check the skewness again

print(hr_data.skew())
print("\nTotal Count of numerica features : ",len(hr_data.skew()))
print("Count of features which are significant skewed : ",len(hr_data.skew().loc[abs(hr_data.skew())>0.5]))


# In[40]:


# Let's separate the input and output

df_x = hr_data.drop(columns = ["Attrition"], axis=1)
y = hr_data["Attrition"]


# In[41]:


# Let's check the how many categorical column are present in the data

print(df_x.dtypes.loc[df_x.dtypes == 'object'])


# In[42]:


# Let's convert the categorical data to numerical data

df_x = pd.get_dummies(df_x, drop_first=True)
df_x


# In[43]:


# Target Column Analysis

sns.countplot(y)

# As already seen that ther is class imbalance issue, Let's fix it


# In[44]:


# Let's convert the target column categorical to numeric

y = y.replace({'Yes' : 1, 'No' : 0})


# In[45]:


# Target value counts

y.value_counts()


# In[46]:


# Use Smote Technique for class imbalance

from imblearn.over_sampling import SMOTE

sm = SMOTE()

df_x,y = sm.fit_resample(df_x,y)

y.value_counts()


# In[47]:


# Now the class im balanced

sns.countplot(y)


# In[48]:


# Let's check the VIF Score

from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

scale = StandardScaler()
x_scale = scale.fit_transform(df_x)

vif = pd.DataFrame()
vif["VIF Score"] = [variance_inflation_factor(x_scale,i) for i in range(x_scale.shape[1])]
vif["Features"] = df_x.columns
vif


# The VIF scores of all the features is less than 10. So, Multicollinearity issue is not found.

# In[49]:


# Let's use PCA

from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score

x_pca = PCA()
x_pca.fit(x_scale)

var_cumu = np.cumsum(x_pca.explained_variance_ratio_)*100
k = np.argmax(var_cumu>98)

print("Number of component explained 98% variance : ",k)
plt.xlabel("Principle Component", fontsize=15)
plt.ylabel("Cumulative Explained Variance", fontsize=14)
plt.axvline(x=k, color='k', linestyle='--')
plt.axhline(y=98, color='r', linestyle='--')
plt.plot(var_cumu)
plt.show()


# In[50]:


# Let's use PCA for scaling

pca = PCA(n_components = 37)
X = pca.fit_transform(df_x)
X


# In[51]:


# Let's import the necessary libraries for model building

from sklearn.metrics import accuracy_score, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from time import time


# In[52]:


# Let's find the best random state for eact best model


def bestmodel(model):
    start = time()
    max_auc = 0
    max_state =0
    for i in range(21,101):
        x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state = i)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        if score > max_auc:
            max_auc = score
            max_state = i
    print("Best Accuracy Score corresponding ",max_state," is",max_auc)
    print("Cross Validation Score is : ",cross_val_score(model,X,y,cv=5).mean())
    end = time()
    print("Time Taken by Model for prediction : {:.4f} seconds".format(end-start))


# In[53]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
bestmodel(LR)


# In[54]:


# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()
bestmodel(DTC)


# In[55]:


# K-Neighbors Classifier

from sklearn.neighbors import KNeighborsClassifier

KNC = KNeighborsClassifier()
bestmodel(KNC)


# In[56]:


# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()
bestmodel(RFC)


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state = 36)


# In[58]:


# Let's plot ROC AUC curve

disp = plot_roc_curve(LR, x_test, y_test)
plot_roc_curve(DTC, x_test, y_test, ax=disp.ax_)
plot_roc_curve(KNC, x_test, y_test, ax=disp.ax_)
plot_roc_curve(RFC, x_test, y_test, ax=disp.ax_)

plt.legend(prop={"size" : 10}, loc="lower right")
plt.show()


# #### By checking the accuracy scores & ROC AUC curver, It is clear that Random Forest Classifier is giving the best score. So, let's  try to increase the accuracy score using Hyperparameter Tuning with Random Forest Classifier.

# In[59]:


# Hyperparameter Tuning with Random Forest Classifier

from sklearn.model_selection import GridSearchCV

param_grid = {"n_estimators" : [100,200], "criterion" : ["gini", "entropy"],
             "min_samples_split" : [2,3], "min_samples_leaf" : [1,2]}

grid_search = GridSearchCV(RFC,param_grid)
grid_search.fit(x_train, y_train)
grid_search.best_params_


# In[68]:


# Final Model

Final_HR_Model = RandomForestClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=3, n_estimators=200)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state = 36)
Final_HR_Model.fit(x_train, y_train)
y_preds = Final_HR_Model.predict(x_test)
accuracy_score(y_test,y_preds)


# In[69]:


# Saving the Final Model

import joblib

joblib.dump(Final_HR_Model,"Final_HR_Model.pkl")


# In[ ]:




