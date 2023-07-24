from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

data = pd.read_csv(r"Deep_Learning\digit-recognizer\train.csv")
y = data['label']
data.drop('label', axis=1, inplace=True)
X = data
y = pd.Categorical(y)


logreg = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
svc = LinearSVC()

model_logreg = logreg.fit(X, y)
model_dt = dt.fit(X, y)
model_svc = svc.fit(X, y)


X_test = pd.read_csv(r"Deep_Learning\digit-recognizer\test.csv")
pred_logreg = model_logreg.predict(X_test)
pred_dt = model_logreg.predict(X_test)
pred_svc = model_logreg.predict(X_test)


pred1 = model_logreg.predict(X)
pred2 = model_dt.predict(X)
pred3 = model_svc.predict(X)
print("Decision Tree Accuracy is: ", accuracy_score(pred1, y)*100)
print("Logistic Regression Accuracy is: ", accuracy_score(pred2, y)*100)
print("Support Vector Machine Accuracy is: ", accuracy_score(pred3, y)*100)
