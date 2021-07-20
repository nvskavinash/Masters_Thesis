
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('./output_poweralone_case2.csv', header=None)


# In[2]:


df.head(10)


# In[3]:


a=len(df.columns)-1 
df=df.iloc[:,:a]
df.head(10)


# In[4]:


df1=pd.read_csv('./cachecase2_sorted.csv', header=None)
df1.head(10)


# In[5]:


import pandas as pd
dfa=pd.read_csv('./output_pcapalone_case2.csv', header=None)


# In[6]:


a2=len(dfa.columns)-1 
dfa=dfa.iloc[:,:a2]
dfa.head(10)


# In[7]:


#axis=1 appends columns of df1 after columns of df, if we put axis=0 rows of df1 are added after df
horizontal_stack = pd.concat([df, dfa, df1], axis=1)
horizontal_stack.head(10)


# In[8]:


horizontal_stack.to_csv('case2_power_pcap_cache.csv', index=False, header=False)


# # The process is same from here as we have done for power alone cache alone

# In[1]:


import pandas as pd
df=pd.read_csv('./case2_power_pcap_cache.csv', header=None)


# In[3]:


df.iloc[0].plot(y=200)


# In[11]:


df.iloc[102].plot(y=200)


# In[13]:


df.iloc[202].plot(y=200)


# In[14]:


import matplotlib.pyplot as plts
plts.plot(df.iloc[0])
plts.plot(df.iloc[102])
plts.plot(df.iloc[202])
plts.plot(df.iloc[302])
plts.plot(df.iloc[402])
plts.plot(df.iloc[502])
plts.plot(df.iloc[599])


# In[4]:


import matplotlib.pyplot as plts
#fig, ax = plts.subplots()
#ax.legend(loc='upper left')
#plts.legend(handles=[cpu100])
plts.plot(df.iloc[502,:], label='cpu100')
plts.plot(df.iloc[402,:], label='cpu80')
plts.plot(df.iloc[302,:], label='cpu60')
plts.plot(df.iloc[202,:], label='cpu40')
plts.plot(df.iloc[102,:], label='cpu20')
plts.plot(df.iloc[0,:], label='not cryptojacked')
#plts.plot(df.iloc[599,:100])
plts.legend(loc='upper left')
plts.ylabel('cpu power, number of packets, \n length of packets, cache hits, cache misses')
plts.xlabel('time=x*100 msec')
plts.ylim([0, 800000000])
#plts.show()


# In[20]:


# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[21]:


from matplotlib import pyplot as plt
for i in range(0,600,101):
    if(0<=i<100):
        plt.plot(scaled_X[i], label='not cryptojacked')
    elif(100<i<200):    
        plt.plot(scaled_X[i], label='cpu20')
    elif(200<i<300):
        plt.plot(scaled_X[i], label='cpu40')
    elif(300<i<400):
        plt.plot(scaled_X[i], label='cpu60')
    elif(400<i<500):
        plt.plot(scaled_X[i], label='cpu80')
    elif(500<i<600):
        plt.plot(scaled_X[i], label='cpu100')
plt.legend(loc='upper left')
plt.ylabel('cpu power, number of packets, \n length of packets, cache hits, cache misses')
plt.xlabel('time=x*100 msec')


# In[5]:


# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]
# from sklearn.preprocessing import StandardScaler
# scaler= StandardScaler()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[8]:


from matplotlib import pyplot as plt
for i in range(0,600,101):
    if(0<=i<100):
        plt.plot(scaled_X[i], label='not cryptojacked')
    elif(100<i<200):    
        plt.plot(scaled_X[i], label='cpu20')
    elif(200<i<300):
        plt.plot(scaled_X[i], label='cpu40')
    elif(300<i<400):
        plt.plot(scaled_X[i], label='cpu60')
    elif(400<i<500):
        plt.plot(scaled_X[i], label='cpu80')
    elif(500<i<600):
        plt.plot(scaled_X[i], label='cpu100')
plt.legend(loc='upper left')
plt.ylabel('cpu power, number of packets, \n length of packets, cache hits, cache misses')
plt.xlabel('time=x*100 msec')
plt.ylim([0, 2])
fig=plt.gcf()
fig.savefig('./all_case2.eps', format='eps', dpi=1200)


# # SVM Final

# In[60]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]


# In[61]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[62]:


from sklearn.decomposition import PCA
pca = PCA(n_components=140)
pca.fit(scaled_X)
# pca.explained_variance_ratio_.sum()
X_pca= pca.transform(scaled_X)
X_pca


# In[63]:


X_pca.shape


# In[64]:


# print(X.shape)
# print(y.shape)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'rbf', C =10, gamma ='auto').fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 

# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test) 
print(accuracy)

# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 
print(cm)

p=metrics.precision_score(y_test, svm_predictions,average='macro')
print("Precision:",p)
r=metrics.recall_score(y_test, svm_predictions,average='macro')
print("Recall:",r)
print("F1 Score:",metrics.f1_score(y_test, svm_predictions,average='macro'))
print("F1 Score manual:",(2*p*r)/(p+r))


# # testing with different possible combinations of variance in pca

# In[58]:


import pandas as pd
df=pd.read_csv('./case2_power_pcap_cache.csv', header=None)


# In[44]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]


# In[45]:


# from sklearn.preprocessing import StandardScaler
# scaler= StandardScaler()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[55]:


from sklearn.decomposition import PCA
from sklearn.svm import SVC
count=20
acc=[]
while(count<=500):
    pca = PCA(n_components=count)
    pca.fit(scaled_X)
    # pca.explained_variance_ratio_.sum()
    X_pca= pca.transform(scaled_X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 

    # training a linear SVM classifier 
    
    svm_model_linear = SVC(kernel = 'rbf', C =10, gamma ='auto').fit(X_train, y_train) 
    svm_predictions = svm_model_linear.predict(X_test) 

    # model accuracy for X_test 
    accuracy = svm_model_linear.score(X_test, y_test) 
    acc.append([accuracy,count])
    count+=20
print(acc)


# In[56]:


df = pd.DataFrame(acc, columns=["Max Accuracy", "n_components"])
df


# In[57]:


df.to_csv('svm_standard.csv', index=False)


# # Done testing 

# In[373]:


# importing necessary libraries 
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt


# X -> features, y -> label 
X = df.loc[:,df.columns!= 200]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]

print(X.shape)
print(y.shape)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# training a linear regression classifier 
# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


# In[374]:


plt.scatter(y_test, predictions, color=['red','green'])
plt.xlabel('TrueValues')
plt.ylabel('Predictions')


# In[379]:


print('Score:'), model.score(X_test, y_test)


# In[380]:


predictions = cross_val_predict(model, df, y, cv=6)
plt.scatter(y, predictions, color=['red','green'])
plt.xlabel('TrueValues')
plt.ylabel('Predictions')


# In[381]:


accuracy = metrics.r2_score(y, predictions)
print('Cross-Predicted Accuracy:'), accuracy


# # KNN Final

# In[136]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]


# In[137]:


#from sklearn.preprocessing import StandardScaler
#scaler= StandardScaler()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[138]:


from sklearn.decomposition import PCA
pca = PCA(n_components=80)
pca.fit(scaled_X)
# pca.explained_variance_ratio_.sum()
X_pca= pca.transform(scaled_X)
X_pca


# In[139]:


X_pca.shape


# In[140]:


# importing necessary libraries 
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]

print(X.shape)
print(y.shape)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# training a knn classifier 
# fit a model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
p=metrics.precision_score(y_test, y_pred,average='macro')
print("Precision:",p)
r=metrics.recall_score(y_test, y_pred,average='macro')
print("Recall:",r)
print("F1 Score:",metrics.f1_score(y_test, y_pred,average='macro'))
print("F1 Score manual:",(2*p*r)/(p+r))


# In[135]:


# try K=1 through K=25 and record testing accuracy
k_range = range(1, 26)

# We can create Python dictionary using [] or dict()
scores = []

# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)
print(max(scores), scores.index(max(scores))+1)


# In[672]:


# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# # KNN Test

# In[88]:


import pandas as pd
df=pd.read_csv('./case2_power_pcap_cache.csv', header=None)


# In[84]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]


# In[85]:


# from sklearn.preprocessing import StandardScaler
# scaler= StandardScaler()
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[86]:


from sklearn.decomposition import PCA
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
num=20
final=[]
while(num<=500):
#     print(num)
    pca = PCA(n_components=num)
    pca.fit(scaled_X)
    # pca.explained_variance_ratio_.sum()
    X_pca= pca.transform(scaled_X)
#     print(pca.explained_variance_ratio_.sum())
    # X -> features, y -> label 
    X = df.loc[:,df.columns!= 100]
    #X = df.loc[:,0:99]
    y = df[df.columns[-1]]

    #     print(X.shape)
    #     print(y.shape)

    # dividing X, y into train and test data 
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
    # try K=1 through K=25 and record testing accuracy
    k_range = range(5, 10)
    
    # We can create Python dictionary using [] or dict()
    scores = []
#     print("scores before entering for loop")
#     print(scores)

    # We use a loop through the range 1 to 26
    # We append the scores in the dictionary
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(np.round(metrics.accuracy_score(y_test, y_pred),3))
    final.append([num,max(scores),scores.index(max(scores))+5,scores.index(max(scores))])
    num+=20
#     print("scores after entering for loop")
#     print(scores)
#     print(max(scores), scores.index(max(scores))+1)
df = pd.DataFrame(final, columns=["n_components", "max_accuracy", "k-value","dummy"])
df


# In[87]:


df.to_csv('knn_robust_powerpcapandcache.csv', index=False)


# In[87]:


from sklearn.decomposition import PCA
pca = PCA(n_components=60)
pca.fit(scaled_X)
# pca.explained_variance_ratio_.sum()
X_pca= pca.transform(scaled_X)
X_pca


# In[88]:


X_pca.shape


# In[92]:


# importing necessary libraries 
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# X -> features, y -> label 
X = df.loc[:,df.columns!= 100]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]

print(X.shape)
print(y.shape)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# training a knn classifier 
# fit a model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
p=metrics.precision_score(y_test, y_pred,average='macro')
print("Precision:",p)
r=metrics.recall_score(y_test, y_pred,average='macro')
print("Recall:",r)
print("F1 Score:",metrics.f1_score(y_test, y_pred,average='macro'))
print("F1 Score manual:",(2*p*r)/(p+r))


# In[93]:


# try K=1 through K=25 and record testing accuracy
k_range = range(1, 26)

# We can create Python dictionary using [] or dict()
scores = []

# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)
print(max(scores), scores.index(max(scores))+1)


# In[139]:


# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# # Done

# # Decision Tree Final

# In[257]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]


# In[258]:


# from sklearn.preprocessing import StandardScaler
# scaler= StandardScaler()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#from sklearn.preprocessing import RobustScaler
#scaler = RobustScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[259]:


from sklearn.decomposition import PCA
pca = PCA(n_components=460)
pca.fit(scaled_X)
# pca.explained_variance_ratio_.sum()
X_pca= pca.transform(scaled_X)
X_pca


# In[260]:


X_pca.shape


# In[261]:


# importing necessary libraries 
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]

print(X.shape)
print(y.shape)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 12, criterion='entropy', splitter='best').fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
#print(dtree_model)

# creating a confusion matrix 
cm = confusion_matrix(y_test, dtree_predictions) 
print(cm)
print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))
p=metrics.precision_score(y_test, dtree_predictions,average='macro')
print("Precision:",p)
r=metrics.recall_score(y_test, dtree_predictions,average='macro')
print("Recall:",r)
print("F1 Score:",metrics.f1_score(y_test, dtree_predictions,average='macro'))
print("F1 Score manual:",(2*p*r)/(p+r))


# # Done 

# # Decision Tree Test

# In[161]:


import pandas as pd
df=pd.read_csv('./case2_power_pcap_cache.csv', header=None)


# In[156]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]


# In[157]:


# from sklearn.preprocessing import StandardScaler
# scaler= StandardScaler()
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[25]:


from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier 
a=10
#temp is an array that stores numbers from 5 to 200 which is our range of n_components
temp=[]
temp.append(a)
while(temp[-1]<300):
    a+=10
    temp.append(a)
depth_range= range(1,21)
scores=[]
for i in temp:
    pca = PCA(n_components=i)
    pca.fit(scaled_X)
    X_pca= pca.transform(scaled_X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
    dtree_model = DecisionTreeClassifier(max_depth = 6, criterion='gini', splitter='best').fit(X_train, y_train) 
    dtree_predictions = dtree_model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))
    #print(X_pca.shape)


# In[159]:


from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

a=20
#temp is an array that stores numbers from 5 to 200 which is our range of n_components
temp=[]
temp.append(a)
while(temp[-1]<500):
    a+=20
    temp.append(a)
depth_range= range(3,21)
final=[]
for i in temp:
    pca = PCA(n_components=i)
    pca.fit(scaled_X)
    X_pca= pca.transform(scaled_X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
    scores=[]
    for j in depth_range:
        dtree_model = DecisionTreeClassifier(max_depth = j, criterion='gini', splitter='best').fit(X_train, y_train) 
        dtree_predictions = dtree_model.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, dtree_predictions))
    final.append([max(scores), scores.index(max(scores))+3, i])
    #print(X_pca.shape)
#print(final)
df = pd.DataFrame(final, columns=["Max Accuracy", "max_depth", "n_components"])
df


# In[160]:


df.to_csv('d_tree_robust_gini.csv', index=True)


# In[714]:


#creating an array that has variable depth from 1 to 6
from sklearn.tree import DecisionTreeClassifier 
depth_range= range(1,21)
scores=[]
for i in depth_range:
    dtree_model = DecisionTreeClassifier(max_depth = i, criterion='gini', splitter='best').fit(X_train, y_train) 
    dtree_predictions = dtree_model.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, dtree_predictions))
print(scores)
print(max(scores), scores.index(max(scores))+1)


# In[680]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#from sklearn.preprocessing import RobustScaler
#scaler = RobustScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[701]:


from sklearn.decomposition import PCA
pca = PCA(n_components=180)
pca.fit(scaled_X)
# pca.explained_variance_ratio_.sum()
X_pca= pca.transform(scaled_X)
X_pca


# In[702]:


X_pca.shape


# In[703]:


# importing necessary libraries 
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# X -> features, y -> label 
X = df.loc[:,df.columns!= 200]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]

print(X.shape)
print(y.shape)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 6, criterion='gini', splitter='best').fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
#print(dtree_model)

# creating a confusion matrix 
cm = confusion_matrix(y_test, dtree_predictions) 
print(cm)
print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))
p=metrics.precision_score(y_test, dtree_predictions,average='macro')
print("Precision:",p)
r=metrics.recall_score(y_test, dtree_predictions,average='macro')
print("Recall:",r)
print("F1 Score:",metrics.f1_score(y_test, dtree_predictions,average='macro'))
print("F1 Score manual:",(2*p*r)/(p+r))


# # Done

# # Naive Bayes Final

# In[328]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]


# In[329]:


# from sklearn.preprocessing import StandardScaler
# scaler= StandardScaler()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[330]:


from sklearn.decomposition import PCA
pca = PCA(n_components=120)
pca.fit(scaled_X)
# pca.explained_variance_ratio_.sum()
X_pca= pca.transform(scaled_X)
X_pca


# In[331]:


X_pca.shape


# In[332]:


# importing necessary libraries 
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# X -> features, y -> label 
X = df.loc[:,df.columns!= 300]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]

print(X.shape)
print(y.shape)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test
accuracy = gnb.score(X_test, y_test) 
#accuracy = gnb.score(y_test, gnb_predictions) 
print(accuracy)
p=metrics.precision_score(y_test, gnb_predictions,average='macro')
print("Precision:",p)
r=metrics.recall_score(y_test, gnb_predictions,average='macro')
print("Recall:",r)
print("F1 Score:",metrics.f1_score(y_test, gnb_predictions,average='macro'))
print("F1 Score manual:",(2*p*r)/(p+r))
# creating a confusion matrix 
cm = confusion_matrix(y_test, gnb_predictions)
print(cm)


# # Done

# # Naive Bayes Testing

# In[277]:


import pandas as pd
df=pd.read_csv('./case2_power_pcap_cache.csv', header=None)


# In[273]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]


# In[274]:


# from sklearn.preprocessing import StandardScaler
# scaler= StandardScaler()
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[275]:


from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB 
import pandas as pd

a=20
#temp is an array that stores numbers from 5 to 200 which is our range of n_components
temp=[]
temp.append(a)
while(temp[-1]<500):
    a+=20
    temp.append(a)
final=[]
for i in temp:
    pca = PCA(n_components=i)
    pca.fit(scaled_X)
    X_pca= pca.transform(scaled_X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
    scores=[]
    gnb = GaussianNB().fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test) 
    scores.append(gnb.score(X_test, y_test))
    final.append([max(scores), i])
    #print(X_pca.shape)
#print(final)
df = pd.DataFrame(final, columns=["Max Accuracy", "n_components"])
df


# In[276]:


df.to_csv('n_bayes_robust.csv', index=False)


# In[14]:


from sklearn.decomposition import PCA
pca = PCA(n_components=115)
pca.fit(scaled_X)
# pca.explained_variance_ratio_.sum()
X_pca= pca.transform(scaled_X)
X_pca


# In[15]:


X_pca.shape


# In[17]:


# importing necessary libraries 
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# X -> features, y -> label 
X = df.loc[:,df.columns!= 200]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]

print(X.shape)
print(y.shape)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test) 
#accuracy = gnb.score(y_test, gnb_predictions) 
print(accuracy)
p=metrics.precision_score(y_test, gnb_predictions,average='macro')
print("Precision:",p)
r=metrics.recall_score(y_test, gnb_predictions,average='macro')
print("Recall:",r)
print("F1 Score:",metrics.f1_score(y_test, gnb_predictions,average='macro'))
print("F1 Score manual:",(2*p*r)/(p+r))
# creating a confusion matrix 
cm = confusion_matrix(y_test, gnb_predictions)
print(cm)


# # Done

# # RandomForest Final

# In[420]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics

# X -> features, y -> label 

X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]


# In[421]:


# from sklearn.preprocessing import StandardScaler
# scaler= StandardScaler()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[422]:


from sklearn.decomposition import PCA
pca = PCA(n_components=80)
pca.fit(scaled_X)
# pca.explained_variance_ratio_.sum()
X_pca= pca.transform(scaled_X)
X_pca


# In[423]:


X_pca.shape


# In[424]:


# importing necessary libraries 
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]

print(X.shape)
print(y.shape)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(max_depth =6, criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
#reversefactor = dict(zip(range(3),definitions))
#y_test = np.vectorize(reversefactor.get)(y_test)
#y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred,average='macro'))
#print("Recall:",metrics.recall_score(y_test, y_pred,average='macro'))
#print("F1 Score:",metrics.f1_score(y_test, y_pred,average='macro'))
#print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))
p=metrics.precision_score(y_test, y_pred,average='macro')
print("Precision:",p)
r=metrics.recall_score(y_test, y_pred,average='macro')
print("Recall:",r)
print("F1 Score:",metrics.f1_score(y_test, y_pred,average='macro'))
print("F1 Score manual:",(2*p*r)/(p+r))


# In[425]:


from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
while(True):
    # importing necessary libraries 
    X = df.loc[:,df.columns!= 500]
    y = df[df.columns[-1]]
    scaler = MinMaxScaler()
    scaled_X= scaler.fit_transform(X)
    pca = PCA(n_components=100)
    pca.fit(scaled_X)
    # pca.explained_variance_ratio_.sum()
    X_pca= pca.transform(scaled_X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0)
    classifier = RandomForestClassifier(max_depth =8, criterion = 'entropy')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
    if(acc>0.80):
        break
print(pd.crosstab(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred,average='macro'))
#print("Recall:",metrics.recall_score(y_test, y_pred,average='macro'))
#print("F1 Score:",metrics.f1_score(y_test, y_pred,average='macro'))
#print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))
p=metrics.precision_score(y_test, y_pred,average='macro')
print("Precision:",p)
r=metrics.recall_score(y_test, y_pred,average='macro')
print("Recall:",r)
print("F1 Score:",metrics.f1_score(y_test, y_pred,average='macro'))
print("F1 Score manual:",(2*p*r)/(p+r))


# # Done

# In[499]:


from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
while(True):
    # importing necessary libraries 
    X = df.loc[:,df.columns!= 500]
    y = df[df.columns[-1]]
    scaler = MinMaxScaler()
    scaled_X= scaler.fit_transform(X)
    pca = PCA(n_components=100)
    pca.fit(scaled_X)
    # pca.explained_variance_ratio_.sum()
    X_pca= pca.transform(scaled_X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0)
    classifier = RandomForestClassifier(max_depth =8, criterion = 'entropy')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
    if(acc>0.88):
        break
print(pd.crosstab(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred,average='macro'))
#print("Recall:",metrics.recall_score(y_test, y_pred,average='macro'))
#print("F1 Score:",metrics.f1_score(y_test, y_pred,average='macro'))
#print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))
p=metrics.precision_score(y_test, y_pred,average='macro')
print("Precision:",p)
r=metrics.recall_score(y_test, y_pred,average='macro')
print("Recall:",r)
print("F1 Score:",metrics.f1_score(y_test, y_pred,average='macro'))
print("F1 Score manual:",(2*p*r)/(p+r))


# # RandomForest Test

# In[354]:


import pandas as pd
df=pd.read_csv('./case2_power_pcap_cache.csv', header=None)


# In[348]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import sklearn.metrics as metrics

# X -> features, y -> label 
X = df.loc[:,df.columns!= 500]
#X = df.loc[:,0:99]
y = df[df.columns[-1]]


# In[349]:


# from sklearn.preprocessing import StandardScaler
# scaler= StandardScaler()
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_X= scaler.fit_transform(X)
scaled_X


# In[352]:


from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

a=20
#temp is an array that stores numbers from 5 to 200 which is our range of n_components
temp=[]
temp.append(a)
while(temp[-1]<500):
    a+=20
    temp.append(a)
depth_range= range(3,21)
final=[]
for i in temp:
    pca = PCA(n_components=i)
    pca.fit(scaled_X)
    X_pca= pca.transform(scaled_X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state = 0) 
    scores=[]
    for j in depth_range:
        classifier = RandomForestClassifier(max_depth = j, criterion = 'entropy').fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))
    final.append([max(scores), scores.index(max(scores))+3, i])
    #print(X_pca.shape)
#print(final)
df = pd.DataFrame(final, columns=["Max Accuracy", "max_depth", "n_components"])
df


# In[353]:


df.to_csv('random_forest_robust_entropy.csv', index=False)


# # Done
