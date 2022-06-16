#import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

y_train = train_data.Survived
X_train = train_data.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived'], axis=1)

#  Reemplazar datos categóricos por 0s y 1s.
X_train['Sex'] = LabelEncoder().fit_transform(X_train['Sex'])

# Imputar valores vacios con el valor medio de la edad
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())

