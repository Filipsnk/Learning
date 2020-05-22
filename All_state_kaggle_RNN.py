from keras import models, layers
from keras.utils import to_categorical

iterations_rnn = {}

mode = input('Choose 0 for standardization and 1 for normalization: ')

for column in full_predictions.columns:
    
    y = y_full[column]
    y = to_categorical(y)
    
    no_of_categories = y.shape[1]
    
    X_train, X_test, y_train, y_test = train_test_split(accepted_train_dummy, y, test_size = 0.25, random_state = 0)
    
    if mode == 0:
        sc = StandardScaler()
    else:
        sc = MinMaxScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    
    model = models.Sequential()
    model.add(layers.Dense(128, activation = 'relu', input_shape = (18,)))
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(128, activation = 'relu')) # Third hidden layer
    model.add(layers.Dense(no_of_categories,activation = 'softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit (X_train,
                         y_train,
                         epochs = 20,
                         batch_size = 1024)
    
    results = model.evaluate(X_test, y_test)
    
    iterations_rnn[column] = round(results[1]*100,2)
    
print(iterations_rnn)
print(iterations_rf)
