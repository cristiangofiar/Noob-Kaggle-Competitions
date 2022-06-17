import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras_tqdm import TQDMCallback
from sklearn.model_selection import train_test_split


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

y_train = train_data.Survived
X_train = train_data.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived'], axis=1)

test_data = test_data.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)

test_data['Sex'] = LabelEncoder().fit_transform(test_data['Sex'])
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

#  Reemplazar datos categóricos por 0s y 1s.
X_train['Sex'] = LabelEncoder().fit_transform(X_train['Sex'])

# Imputar valores vacios con el valor medio de la edad
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())


def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    keras.layers.Dense(1, activation='sigmoid')
  ])

  optimizer = keras.optimizers.SGD(0.01)

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  return model

model = build_model()

model.summary()

#X_train, y_train= train_test_split(X_train, y_train,random_state=32)

model.fit(X_train, y_train, epochs=150, validation_split=0.2)

# Generar predicciones
predictions = model.predict(test_data)

predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

prediction = pd.DataFrame({'PassengerId': test_data.PassengerId.values, 'Survived': predictions.ravel().astype('int64')})
prediction.to_csv('data/initial_submission.csv', index=False)


