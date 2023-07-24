import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

input_data = pd.read_csv(r"Deep_Learning\digit-recognizer\train.csv")


y = input_data['label']
input_data.drop('label', axis=1, inplace=True)
X = input_data
# converts categorical variables into a set of binary variables
y = pd.get_dummies(y)

classifier = Sequential()
classifier.add(Dense(units=600, kernel_initializer='uniform',
               activation='relu', input_dim=784))
classifier.add(
    Dense(units=400, kernel_initializer='uniform', activation='relu'))
classifier.add(
    Dense(units=200, kernel_initializer='uniform', activation='relu'))
classifier.add(
    Dense(units=10, kernel_initializer='uniform', activation='sigmoid'))


classifier.compile(
    optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X, y, batch_size=10, epochs=5)

test_data = pd.read_csv(r"Deep_Learning\digit-recognizer\test.csv")
y_pred = classifier.predict(test_data)
