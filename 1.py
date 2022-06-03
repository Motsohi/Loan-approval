#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sysconfig import get_python_version
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_python_version().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("Data Group 4.csv")
df.head(10)


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.info()


# In[7]:


#misssing values from each column
df.isnull().sum()


# In[8]:


#find the total number of missing values from the whole/entire dataset
df.isnull().sum().sum()


# In[9]:


df.mean()


# In[10]:


#using mode to handle missing values
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df.isnull().sum()


# In[11]:


df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())


# In[12]:


df.isnull().sum()


# In[13]:


cols=['Dependents']
df.loc[:,cols]=df.loc[:,cols].ffill()
print(df.head())


# In[14]:


df.isnull().sum()


# In[15]:


cols=['Credit_History']
df.loc[:,cols]=df.loc[:,cols].ffill()
print(df.head())


# In[16]:


df.isnull().sum()


# In[17]:


df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].median())


# In[18]:


df.isnull().sum()


# In[19]:


df=df.dropna(axis=0)


# In[20]:


df.isnull().sum()


# In[21]:


#identifying outliers
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['ApplicantIncome'])
plt.subplot(1,2,2)
sns.distplot(df['CoapplicantIncome'])
plt.show()


# In[22]:


import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['LoanAmount'])
plt.subplot(1,2,2)
sns.distplot(df['Loan_Amount_Term'])
plt.show()


# In[23]:


import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['Credit_History'])
plt.show()


# In[24]:


print("Highest allowed",df['ApplicantIncome'].mean()+3*df['ApplicantIncome'].std())
print("Lowest allowed",df['ApplicantIncome'].mean()-3*df['ApplicantIncome'].std())


# In[25]:


#finding outliers
df[(df['ApplicantIncome']>24695.80)|(df['ApplicantIncome']<-13625.74)]


# In[26]:


print("Highest allowed",df['Loan_Amount_Term'].mean()+3*df['Loan_Amount_Term'].std())
print("Lowest allowed",df['Loan_Amount_Term'].mean()-3*df['Loan_Amount_Term'].std())


# In[27]:


#finding outliers
df[(df['Loan_Amount_Term']>531.58)|(df['Loan_Amount_Term']<155.24)]


# In[28]:


print("Highest allowed",df['CoapplicantIncome'].mean()+3*df['CoapplicantIncome'].std())
print("Lowest allowed",df['CoapplicantIncome'].mean()-3*df['CoapplicantIncome'].std())


# In[29]:


#finding outliers
df[(df['CoapplicantIncome']>8258.69)|(df['CoapplicantIncome']<-5165.01)]


# In[30]:


print("Highest allowed",df['LoanAmount'].mean()+3*df['LoanAmount'].std())
print("Lowest allowed",df['LoanAmount'].mean()-3*df['LoanAmount'].std())


# In[31]:


#finding outliers
df[(df['LoanAmount']>384.40)|(df['LoanAmount']<-96.22)]


# In[32]:


print("Highest allowed",df['Credit_History'].mean()+3*df['Credit_History'].std())
print("Lowest allowed",df['Credit_History'].mean()-3*df['Credit_History'].std())


# In[33]:


#finding outliers
df[(df['Credit_History']>1.92)|(df['Credit_History']<-0.21)]


# In[34]:


#Handling outliers
#Trimming
new_df=df[(df['ApplicantIncome']<24695.80)&(df['ApplicantIncome']>-13625.74)]
new_df


# In[35]:


#Capping Outliers
upper_limit=df['ApplicantIncome'].mean()+3*df['ApplicantIncome'].std()
lower_limit=df['ApplicantIncome'].mean()-3*df['ApplicantIncome'].std()


# In[36]:


#Apply the capping
df['ApplicantIncome']=np.where(
df['ApplicantIncome']>upper_limit,upper_limit,np.where(df['ApplicantIncome']<lower_limit,lower_limit,df['ApplicantIncome'])
)


# In[37]:


#Describe
df['ApplicantIncome'].describe()


# In[38]:


#Handling outliers
#Trimming
new_df=df[(df['CoapplicantIncome']<8258.69)&(df['CoapplicantIncome']>-5165.01)]
new_df


# In[39]:


#Capping Outliers
upper_limit=df['CoapplicantIncome'].mean()+3*df['CoapplicantIncome'].std()
lower_limit=df['CoapplicantIncome'].mean()-3*df['CoapplicantIncome'].std()


# In[40]:


#Apply the capping
df['CoapplicantIncome']=np.where(
df['CoapplicantIncome']>upper_limit,upper_limit,np.where(df['CoapplicantIncome']<lower_limit,lower_limit,df['CoapplicantIncome'])
)


# In[41]:


#Describe
df['CoapplicantIncome'].describe()


# In[42]:


#Handling outliers
#Trimming
new_df=df[(df['LoanAmount']<384.40)&(df['LoanAmount']>-96.22)]
new_df


# In[43]:


#Capping Outliers
upper_limit=df['LoanAmount'].mean()+3*df['LoanAmount'].std()
lower_limit=df['LoanAmount'].mean()-3*df['LoanAmount'].std()


# In[44]:


#Apply the capping
df['LoanAmount']=np.where(
df['LoanAmount']>upper_limit,upper_limit,np.where(df['LoanAmount']<lower_limit,lower_limit,df['LoanAmount'])
)


# In[45]:


#Describe
df['LoanAmount'].describe()


# In[46]:



#Handling outliers
#Trimming
new_df=df[(df['Loan_Amount_Term']<531.58)&(df['Loan_Amount_Term']>155.24)]
new_df


# In[47]:


#Capping Outliers
upper_limit=df['Loan_Amount_Term'].mean()+3*df['Loan_Amount_Term'].std()
lower_limit=df['Loan_Amount_Term'].mean()-3*df['Loan_Amount_Term'].std()


# In[48]:


#Apply the capping
df['Loan_Amount_Term']=np.where(
df['Loan_Amount_Term']>upper_limit,upper_limit,np.where(df['Loan_Amount_Term']<lower_limit,lower_limit,df['Loan_Amount_Term'])
)


# In[49]:


#Describe
df['Loan_Amount_Term'].describe()


# In[50]:


#Descri0ptive Analysis
df.describe()


# In[51]:


df.median()


# In[52]:


df.skew()


# In[53]:


df.shape


# In[54]:


df.dtypes


# In[55]:


df.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[56]:


df.head()


# In[57]:


#dependents column
df['Dependents'].value_counts()


# In[58]:


#change 3+ to 4
df=df.replace(to_replace='3+',value=4)


# In[59]:


df['Dependents'].value_counts()


# In[60]:


#education and loan status
sns.countplot(x='Education', hue='Loan_Status',data=df)


# In[61]:


#marital status and loan status
sns.countplot(x='Married', hue='Loan_Status',data=df)


# In[62]:


#converting cataegorical columns to numeric values
df.replace({"Married":{'No':0,'Yes':1},"Gender":{'Male':1,'Female':0},"Self_Employed":{'No':0,'Yes':1},"Property_Area":{'Rural':0,'Semiurban':1,'Urban':2},"Education":{'Graduate':1,'Not Graduate':0}}, inplace=True)


# In[63]:


df.head()


# In[64]:


X=df.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y=df['Loan_Status']


# In[65]:


print(X)
print(Y)


# In[66]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)


# In[68]:


print(X.shape,X_train.shape,X_test.shape)


# In[69]:


#creating model
classifier=svm.SVC(kernel='linear')


# In[70]:


#training model
classifier.fit(X_train,Y_train)


# In[71]:


#model evaluation
#accuracy score
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[72]:


print('Accuracy on training data:',training_data_accuracy)


# In[73]:


#accuracy score
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[74]:


print('Accuracy on testing data:',test_data_accuracy)


# In[ ]:




