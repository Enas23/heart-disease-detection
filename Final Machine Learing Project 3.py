#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import scipy as sp
import seaborn as sns 
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.pipeline import Pipeline
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import datasets
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("D:/Collage/Heart_Disease_Prediction.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.size


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.shape


# In[9]:


df.isnull()


# In[10]:


df.isnull().sum()


# In[11]:


# 1st way to Visualize features of data 
df.plot()


# In[12]:


#It is always better to check the correlation between the features so that we can analyze(positive or Negative)
plt.figure(figsize=(20,12))
sns.set_context('notebook',font_scale = 1.3)
sns.heatmap(df.corr(),annot=True,linewidth =2)
plt.tight_layout()


# In[13]:


df['Heart Disease'].value_counts()


# In[14]:


x=df.drop(columns='Heart Disease',axis=1)
y=df['Heart Disease']


# In[15]:


print(x)


# In[16]:


print(y)


# In[27]:


#Descision Tree Algorithm
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
dt=DecisionTreeClassifier(max_depth=6,random_state=1)
dt.fit(x_train,y_train) #takes the training data as arguments
y_pred=dt.predict(x_test)
accuracy_for_descisiontree=accuracy_score(y_test,y_pred)*100
print("the accuracy for descisiontree: {:.1f}".format(accuracy_for_descisiontree))


# In[18]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
x,y = make_classification(random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
clf = SVC(random_state=42)
clf.fit(x_train,y_train)
SVC(random_state=42)
plot_confusion_matrix(clf,x_test,y_test)  
plt.show()


# In[19]:


#2ed KNN Algorithm
df["Heart Disease"]=df["Heart Disease"].map({'Absence':0,'Presence':1}).astype(int)
x=df.drop(columns='Heart Disease',axis=1)
y=df["Heart Disease"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
training_acc=[]
testing_acc=[]
k=range(1,100,2)
score=0
for i in k:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train) #takes the training data as arguments
y_pred_train=knn.predict(x_train)
training_acc.append(accuracy_score(y_train,y_pred_train))
y_pred_test=knn.predict(x_test)
acc=accuracy_score(y_pred_test,y_test)
acc=accuracy_score(y_pred_test,y_test)*100
if score<acc:
 score=acc
 best_k=i
print(i)
print(score)
print(acc)


# In[20]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
x,y = make_classification(random_state=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
clf = SVC(random_state=1)
clf.fit(x_train,y_train)
SVC(random_state=1)
plot_confusion_matrix(clf,x_test,y_test)  
plt.show()


# In[22]:


#perciptron
x=df.drop(columns='Heart Disease',axis=1)
y=df["Heart Disease"]
class_le=LabelEncoder()
y=class_le.fit_transform(df['Heart Disease'].values)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.32)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
scaler=StandardScaler()

print('\nData preprocessing with {scaler}\n'.format(scaler=scaler))

x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)
mlp=MLPClassifier(
    max_iter=1000,
    alpha=0.1,

    random_state=42
)
mlp.fit(x_train_scaler,y_train )

mlp_predict = mlp.predict(x_test_scaler)
MLP_Accuracy=accuracy_score(y_test, mlp_predict) * 100
print('MLP_Accuracy: {:.2f}%'.format(accuracy_score(y_test, mlp_predict) * 100))


# In[23]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
x,y = make_classification(random_state=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
clf = SVC(random_state=42)
clf.fit(x_train,y_train)
SVC(random_state=42)
plot_confusion_matrix(clf,x_test,y_test)  
plt.show()


# In[24]:


#Bonus Reinforcement Learing
x=df.drop(columns='Heart Disease',axis=1)
y=df['Heart Disease']

#Bonus Reinforcement learing
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import datasets
# Using Random forest classifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)
Accuracy_of_Random_Forest_Classifier=metrics.accuracy_score(y_test, y_pred_rf)*100
print("Accuracy of Random Forest Classifier :: ", metrics.accuracy_score(y_test, y_pred_rf)*100)


# In[25]:


import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_axes([1,1,1,1])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Comparision between Algorithms Shown in Graph")
Algorithms=['Random_Forest','Presciptron', 'descision tree', 'knn']
Accuracy=[Accuracy_of_Random_Forest_Classifier,MLP_Accuracy,accuracy_for_descisiontree,acc]
ax.bar(Algorithms,Accuracy)
plt.show()


# In[ ]:




