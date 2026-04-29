# %%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib



ingest = pd.read_csv('data/amazon.csv')


# %%
ingest = ingest.dropna()
# %%
ingest.head()

# %%
ingest['discounted_price']= ingest['discounted_price'].str.replace('₹','').str.replace(',','').astype(float)
ingest['actual_price']= ingest['actual_price'].str.replace('₹','').str.replace(',','').astype(float)
ingest['discount_percentage'] = ingest['discount_percentage'].str.replace('%','').astype(float)
ingest['rating_count'] = ingest['rating_count'].str.replace(',','').astype(float)

ingest.head()
# %%
ingest['rating'] = pd.to_numeric(ingest['rating'], errors='coerce')

ingest['label']= ingest['rating'].apply(lambda x: 1 if x>=4 else 0)

ingest['label'].value_counts()
# %%
ingest['price_diff'] = ingest['actual_price'] - ingest['discounted_price']
ingest['discounted_ratio'] = ingest['discounted_price'] / ingest['actual_price']
ingest['log_rating_count'] = np.log(ingest['rating_count'])

ingest.head()
# %%
X = ingest[['discounted_price','discounted_ratio','log_rating_count','price_diff','actual_price']]
Y = ingest['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# %%
X.isnull().sum()
# %%
len(ingest
    ) #ok so there are a total of 1465 rows ...there is 2 rows with Nan value..so i am dropping them..coz it will only be 0.13% of missing data
# %%
model1 = LogisticRegression(max_iter=5000)
model1.fit(X_train, Y_train)
# %%
oy_valpred = model1.predict(X_test)
print('Accuracy is: ', accuracy_score(Y_test, y_valpred)*100,'%')
print('Confusion :', confusion_matrix(Y_test, y_valpred))
# %%
#75% is kinda okish but not so satisfactory
# %%
#yeah so i used logistic regression for it but for the 2nd model might check with knn decision tree

# %%
joblib.dump(model1, 'model1.pkl')