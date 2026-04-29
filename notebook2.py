# %%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# %%
model1 = joblib.load("model1.pkl")
# %%
ingest = pd.read_csv('data/amazon.csv')
# %%
ingest = ingest.dropna()
# %%
ingest.head()
# %%
ingest['discounted_price']= ingest['discounted_price'].str.replace('₹','').str.replace(',','').astype(float)
ingest['actual_price']=ingest['actual_price'].str.replace('₹','').str.replace(',','').astype(float)
ingest['discount_percentage'] = ingest['discount_percentage'].str.replace('%','').astype(float)
ingest['rating_count']= ingest['rating_count'].str.replace(',','').astype(float)

ingest.head()
# %%
ingest['rating'] = pd.to_numeric(ingest['rating'], errors='coerce')
ingest['label']=ingest['rating'].apply(lambda x:1 if x>=4 else 0)
ingest['label'].value_counts()
# %%
X = ingest[['discounted_price','actual_price','rating_count']]
Y = ingest['label']
# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
# %%
X.isnull().sum()
# %%
len(ingest)
# %%
ingest['price_diff'] = ingest['actual_price'] - ingest['discounted_price']
ingest['discounted_ratio'] = ingest['discounted_price'] / ingest['actual_price']
ingest['log_rating_count'] = np.log(ingest['rating_count'])

ingest.head()
# %%
X2 = ingest[['discounted_price','discounted_ratio','log_rating_count','price_diff','actual_price']]
Y2 = ingest['label']

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size = 0.2, random_state = 42)
# %%
rf = RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_split=5,min_samples_leaf=2,random_state=42)
rf.fit(X_train2, Y_train2)

Y_testpred_rf=rf.predict(X_test2)

print("Test Accuracy:",accuracy_score(Y_test2, Y_testpred_rf) * 100, "%")

# %%
rf_grid = {'n_estimators':[100,200,250, 280, 300,400],'max_depth': [5,7,8,9,10,None],'min_samples_split':[2,3,4,5], 'min_samples_leaf':[1,2,3]}

rf = RandomForestClassifier(random_state=42)

grid= GridSearchCV(
    estimator=rf, param_grid=rf_grid,cv=4, scoring='accuracy', n_jobs=-1)
grid.fit(X_train2, Y_train2)

print("Best  Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
print("Accuracy:", grid.score(X_test2, Y_test2)*100, "%")
# %%
joblib.dump(rf, 'model2.pkl')
# %%
model2 = joblib.load("model2.pkl")
# %%
model2.fit(X_train2, Y_train2)
# %%
Y_prediction1 = model1.predict(X_test2)
Y_prediction2 = model2.predict(X_test2)
# %%
acc1 = accuracy_score(Y_test2, Y_prediction1)*100
prec1 = precision_score(Y_test2, Y_prediction1)*100
f11 = f1_score(Y_test2, Y_prediction1)*100
# %%
acc2 = accuracy_score(Y_test2, Y_prediction2)*100
prec2 = precision_score(Y_test2, Y_prediction2)*100
f12 = f1_score(Y_test2, Y_prediction2)*100
# %%
print("Comparison \n")
print("Logistic Regression: \nAccuracy: ", acc1, "%\n")
print("Precision: ", prec1, "%\n")
print("F1 Score: ", f11, "%\n")
# %%
print("Comparison \n")
print("Random Forest \nAccuracy: ", acc2, "%\n")
print("Precision: ", prec2, "%\n")
print("F1 Score: ", f12, "%\n")