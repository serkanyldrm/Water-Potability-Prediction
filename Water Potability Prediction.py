#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# ## Importing Data

# In[2]:


potability_matrix = pd.read_csv('drinking_water_potability.csv')
potability_matrix = potability_matrix.apply (pd.to_numeric, errors='coerce')
potability_matrix_clean = potability_matrix.dropna()


# ## Data Analysis

# In[ ]:


potability_matrix.Potability.value_counts()


# In[ ]:



pot_lbl = potability_matrix.Potability.value_counts()

plt.figure(figsize=(8,5))
sns.barplot(pot_lbl.index, pot_lbl)
plt.xlabel('Potability',fontsize=15)
plt.ylabel('count',fontsize=15)


# In[ ]:


potability_matrix_clean.Potability.value_counts()


# In[ ]:


pot_lbl = potability_matrix_clean.Potability.value_counts()

plt.figure(figsize=(8,5))
sns.barplot(pot_lbl.index, pot_lbl)
plt.xlabel('Potability',fontsize=15)
plt.ylabel('count',fontsize=15)


# Correlation Matrix

# In[ ]:


x = potability_matrix.drop("Potability", axis = 1)
y = potability_matrix["Potability"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
plt.figure(figsize=(10,10))
cor = x_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.savefig('correlation.png', dpi=1000)
plt.show()


# Box-Plot

# In[ ]:


x_cols = ['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes', 'Turbidity']
plt.figure(figsize=(20,20))
for col in x_cols:
    plt.figure()
    #plt.subplot(3,3,i+1)
    potability_matrix.boxplot([col])


# ## Preprocessing

# Missing Values

# In[ ]:


ph_potab0 = potability_matrix[potability_matrix['Potability']==0]['ph'].mean()
ph_potab1 = potability_matrix[potability_matrix['Potability']==1]['ph'].mean()

sulfate_potab0 = potability_matrix[potability_matrix['Potability']==0]['Sulfate'].mean()
sulfate_potab1 =potability_matrix[potability_matrix['Potability']==1]['Sulfate'].mean()

tri_potab0 = potability_matrix[potability_matrix['Potability']==0]['Trihalomethanes'].mean()
tri_potab1 = potability_matrix[potability_matrix['Potability']==1]['Trihalomethanes'].mean()


# In[ ]:


potability_matrix.loc[(potability_matrix['Potability'] == 0) & (potability_matrix['ph'].isna()),'ph'] = ph_potab0 
potability_matrix.loc[(potability_matrix['Potability'] == 1) & (potability_matrix['ph'].isna()),'ph'] = ph_potab1

potability_matrix.loc[(potability_matrix['Potability'] == 0) & (potability_matrix['Sulfate'].isna()),'Sulfate'] = sulfate_potab0
potability_matrix.loc[(potability_matrix['Potability'] == 1) & (potability_matrix['Sulfate'].isna()),'Sulfate'] = sulfate_potab1

potability_matrix.loc[(potability_matrix['Potability'] == 0) & (potability_matrix['Trihalomethanes'].isna()),'Trihalomethanes'] = tri_potab0
potability_matrix.loc[(potability_matrix['Potability'] == 1) & (potability_matrix['Trihalomethanes'].isna()),'Trihalomethanes'] = tri_potab1


# Outliers with Z-Scores 

# In[ ]:


x_cols = ['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes', 'Turbidity']
delete = []
for col in x_cols:
    z_scores = zscore(potability_matrix[col])
    abs_z_scores = np.abs(z_scores)
    count = 0
    potability_matrix_np = potability_matrix.values
    for i in range(abs_z_scores.shape[0]):
        if((abs_z_scores[i] > 3)):
            count += 1
            delete.append(i) 
    potability_matrix_np = np.delete(potability_matrix_np, delete, axis=0)
potability_matrix = pd.DataFrame(potability_matrix_np, columns = ['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes', 'Turbidity','Potability'])         


# In[ ]:


potability_matrix


# Outliers with Z-Scores for Clean Data

# In[ ]:


x_cols = ['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes', 'Turbidity']
delete = []
for col in x_cols:
    z_scores = zscore(potability_matrix_clean[col])
    abs_z_scores = np.abs(z_scores)
    count = 0
    potability_matrix_clean_np = potability_matrix_clean.values
    for i in range(abs_z_scores.shape[0]):
        if((abs_z_scores[i] > 3)):
            count += 1
            delete.append(i) 
    potability_matrix_clean_np = np.delete(potability_matrix_clean_np, delete, axis=0)
potability_matrix_clean = pd.DataFrame(potability_matrix_clean_np, columns = ['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes', 'Turbidity','Potability'])         


# In[ ]:


potability_matrix_clean


# ## Models for Raw Data with Mean Values (Decision tree, Naive Bayes, Random Forest)

# In[ ]:


x = potability_matrix.drop("Potability", axis = 1)
y = potability_matrix["Potability"]
models = [DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier()]


# In[ ]:


Acc = []
decisionTree = []
naiveBayes = []
randomForest = []
w, h = 2, 3
Matrix_raw = [[0 for x in range(w)] for y in range(h)] 
count = 0
for i in models:
    for j in range(100):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        i = i.fit(x_train, y_train)
        y_pred = i.predict(x_test)
        Acc.append(accuracy_score(y_test, y_pred))
    Matrix_raw[count][0] = y_pred
    Matrix_raw[count][1] = y_test
    count += 1
for i in range(100):
    decisionTree.append(Acc[i])
    naiveBayes.append(Acc[i+100])
    randomForest.append(Acc[i+200])
plt.plot(decisionTree,'r',label = 'Decision Tree')
plt.plot(naiveBayes,'g',label = 'Naive Bayes')
plt.plot(randomForest,'b',label = 'Random Forest')
plt.legend(loc="lower right", fontsize=6)
plt.ylim(0,1)
plt.ylabel('Accuracy')
plt.title('Raw Data with Mean Values')
plt.savefig("Raw Data with Mean Values.png",dpi=600)
plt.show()


# ## Models for Clean Data (Decision tree, Naive Bayes, Random Forest)

# In[ ]:


x_clean = potability_matrix_clean.drop("Potability", axis = 1)
y_clean = potability_matrix_clean["Potability"]


# In[ ]:


Acc = []
decisionTree = []
naiveBayes = []
randomForest = []
w, h = 2, 3
Matrix_clean = [[0 for x in range(w)] for y in range(h)] 
count = 0
for i in models:
    for j in range(100):
        x_train_clean, x_test_clean, y_train_clean, y_test_clean = train_test_split(x_clean, y_clean, test_size=0.2, random_state=0)
        i = i.fit(x_train_clean, y_train_clean)
        y_pred_clean = i.predict(x_test_clean)
        Acc.append(accuracy_score(y_test_clean, y_pred_clean))
    Matrix_clean[count][0] = y_pred_clean
    Matrix_clean[count][1] = y_test_clean
    count +=1
for i in range(100):
    decisionTree.append(Acc[i])
    naiveBayes.append(Acc[i+100])
    randomForest.append(Acc[i+200])
plt.plot(decisionTree,'r',label = 'Decision Tree')
plt.plot(naiveBayes,'g',label = 'Naive Bayes')
plt.plot(randomForest,'b',label = 'Random Forest')
plt.legend(loc="lower right", fontsize=6)
plt.ylim(0,1)
plt.ylabel('Accuracy')
plt.title('Clean Data')
plt.savefig('Clean Data.png',dpi=600)
plt.show()


# ## Comparing Models

# Confusing Matrix and F-scores for Raw Data with Mean Values

# In[ ]:


for a in range(0,3):
    conf_matrix = confusion_matrix(y_true=Matrix_raw[a][1], y_pred=Matrix_raw[a][0])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    f1=f1_score(Matrix_raw[a][1], Matrix_raw[a][0])
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('F-score:'+str(f1), fontsize=18)
    plt.show()


# Confusing Matrix and F-scores for Clean Data

# In[ ]:


for a in range(0,3):   
    conf_matrix = confusion_matrix(y_true=Matrix_clean[a][1], y_pred=Matrix_clean[a][0])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    f1=f1_score(Matrix_clean[a][1], Matrix_clean[a][0])
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('F-score:'+str(f1), fontsize=18)
    plt.show()

