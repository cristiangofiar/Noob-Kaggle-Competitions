from tensorflow import keras
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('data/train.csv')
_test_data = pd.read_csv('data/test.csv')

features_to_remove = ['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

y_train = train_data.Survived

X_train = train_data.drop(features_to_remove + ['Survived'], axis=1)
#  Reemplazar datos categóricos por 0s y 1s.
X_train['Sex'] = LabelEncoder().fit_transform(X_train['Sex'])
# Imputar valores vacios con el valor medio de la edad
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
# convertir columna de datos categóricos a varios categorias numéricas
X_train = pd.get_dummies(X_train, columns=['Pclass'], prefix='Pclass') 

#lo mismo con test_data
test_data = _test_data.drop(features_to_remove,axis=1)
test_data['Sex'] = LabelEncoder().fit_transform(test_data['Sex'])
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data = pd.get_dummies(test_data, columns=['Pclass'], prefix='Pclass') 


def build_model():
  model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=[len(X_train.keys())]),
   # keras.layers.Dropout(0.2), 
    keras.layers.Dense(512, activation='relu'),
   # keras.layers.Dropout(0.2),  
    keras.layers.Dense(512, activation='relu'),
   # keras.layers.Dropout(0.2),  
    keras.layers.Dense(128, activation='relu'),
   # keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid') 
  ])

  model.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=0.003),
                metrics=['accuracy'])

  return model

model = build_model()
model.summary()

history = model.fit(
    X_train,
    y_train,
    verbose=2, epochs=100,batch_size=5,validation_split=0.2)

# Generar predicciones
predictions = model.predict(test_data)

predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

prediction = pd.DataFrame({'PassengerId': _test_data.PassengerId.values, 'Survived': predictions.ravel().astype('int64')})
prediction.to_csv('data/initial_submission.csv', index=False)

resultados = pd.read_csv('data/initial_submission.csv')
print(resultados.head())


