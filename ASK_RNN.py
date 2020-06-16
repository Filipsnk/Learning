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
    model.add(layers.Dense(128, activation = 'relu'))
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

iterations_rnn_five_layers = iterations_rnn

iterations_rnn_standarization = iterations_rnn
iterations_rnn_normalization = iterations_rnn

print(iterations_rnn_normalization)
print(iterations_rnn_standarization)

mean_accuracy(iterations_rnn_five_layers)

def mean_accuracy(dataset):

    avg_accuracy = 0
    
    for acc in dataset.values():
        avg_accuracy = avg_accuracy + acc
    
    avg_accuracy = avg_accuracy / len(dataset.values())
    
    print(avg_accuracy)

## normalization - 60,34%
## standarization - 59,94%
## five layers - 60,55%

### 

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from numpy import mean, std

y = y_full['A']
model = GradientBoostingClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, accepted_train_dummy, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))