from keras import models, layers

for column in pred_columns:
    del accepted_train_dummy[column] 

y = y_full['A']
X_train, X_test, y_train, y_test = train_test_split(accepted_train_dummy, y, test_size = 0.25, random_state = 0)

model = models.Sequential()
model.add(layers.Dense(16, activation = 'tanh', input_shape = (28,)))
model.add(layers.Dense(16, activation = 'tanh'))
model.add(layers.Dense(16, activation = 'tanh')) # Third hidden layer
model.add(layers.Dense(1,activation = 'sigmoid'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])

history = model.fit (X_train,
                     y_train,
                     epochs = 4,
                     batch_size = 512)

results5 = model.evaluate(X_test, y_test)
