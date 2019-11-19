from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

breast_cancer = load_breast_cancer()

X=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
Y=pd.Categorical.from_codes(breast_cancer.target,breast_cancer.target_names)

encoder = LabelEncoder()
binary_encoded_y = pd.Series(encoder.fit_transform(Y))

train_X,test_X,train_Y,test_Y = train_test_split(X,binary_encoded_y,random_state=1)

classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)

classifier.fit(train_X,train_Y)

predictions = classifier.predict(test_X)



print(confusion_matrix(test_Y,predictions))
