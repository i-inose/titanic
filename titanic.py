import pandas as pd
titanic=pd.read_csv('titanic.csv')
titanic=titanic.drop(['name','row.names'],axis=1)
mean=round(titanic['age'].mean(),2)
titanic['age'].fillna(mean,inplace=True)
titanic.fillna("",inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in titanic.columns.values.tolist():
 if (i=='age'):
  pass
 else:
  titanic[i] = le.fit_transform(titanic[i])

trees=493
crossv=10
X = titanic.drop(["survived"], axis=1)
y = titanic["survived"]
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=crossv, test_size=0.1, random_state=10)
clf = RandomForestClassifier(criterion="entropy", n_estimators=trees, max_depth=None, min_samples_split=2, random_state=10, n_jobs=-1)
scores = cross_val_score(clf, X, y, cv=cv)
print("scores: ", scores)
print("scores mean: ", scores.mean())
print("scores std: ", round(scores.std(), 6))
