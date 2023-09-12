# Wine-Quality-Prediction-using-Logistic-Regression-Status

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
df = pd.read_csv("../input/wine-quality-binary-classification/wine.csv")
df.head()
df.info()
# Encoding categorical variable
df['quality_cat'] = df['quality'].astype('category').cat.codes
df.head()
sns.pairplot(df,hue='quality')
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df.hist(ax = ax)
df.quality_cat.value_counts().plot(kind='bar')
plt.xlabel("Good or Bad")
plt.ylabel("Count")
plt.title("Quality")
#Here we can see that dataset is not much imbalanced so there is no need to balance.
lt.figure(figsize=(20,10))
plt.subplots_adjust(left=0, bottom=0.5, right=0.9, top=0.9, wspace=0.5, hspace=0.8)
plt.subplot(141)
plt.title('Percentage of good and bad quality wine',fontsize = 20)
df['quality'].value_counts().plot.pie(autopct="%1.1f%%")

df1 = df.drop('quality',axis=1)
df1.info()
X = df1.drop('quality_cat',axis=1)
Y = df1['quality_cat']

Y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2,random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

Implementing logistic Regression Using Sklearn
from sklearn.linear_model import LogisticRegression # for Logistic Regression Algorithm
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train ,y_train)
y_pred = lr.predict(X_test)
lr.score(X_test,y_test)
from sklearn.metrics import accuracy_score
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)
THRESHOLD = 0.5
y_pred = np.where(y_pred>0.5,1,0)
y_pred



Implementing Logistic Regression from Scratch
lr = 0.06

# Parameters initialization
weights = np.random.normal(0, 0.1, 11)
biais = random.normalvariate(0, 0.1)

m = X_train.shape[0]
for epoch in range(1000):

    # Forward pass
    Z = np.dot(X_train, weights) + biais
    A = 1 / (1 + np.exp(-Z))
    
    #Loss Computation
    J = np.sum(-(y_train * np.log(A) + (1 - y_train) * np.log(1 - A))) / m

    # Gradient computation
    dZ = A - y_train
    dw = np.dot(dZ, X_train) / m
    db = np.sum(dZ) / m

    # Update weights
    weights = weights - lr * dw
    biais = biais - lr * db
    
    if epoch % 10 == 0:
        print("epoch %s - loss %s" % (epoch, J))



        preds = []
for feats in X_test:

    z = np.dot(feats, weights) + biais
    a = 1 / (1 + np.exp(-z))

    if a > 0.5:
        preds.append(1)
    elif a <= 0.5:
        preds.append(0)
        from sklearn.metrics import classification_report
target_names = ['Bad', 'Good']
print(classification_report(y_test, preds, target_names=target_names))
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(mat,xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'], fmt='.0f',annot=True)
from sklearn.metrics import f1_score
f1 = f1_score(y_test, preds)
print('F1 score: %f' % f1)

from sklearn.metrics import precision_score, recall_score
print('Precision is: ', precision_score(y_test, preds))
print('Recall is: ', recall_score(y_test, preds))
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print('True negatives: ', tn, '\nFalse positives: ', fp, '\nFalse negatives: ', fn, '\nTrue Positives: ', tp)
