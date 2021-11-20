from dbn.tensorflow import SupervisedDBNClassification
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

digits = pd.read_csv("train.csv")

X = np.array(digits.drop(["label"], axis=1))
Y = np.array(digits["label"])

from sklearn.preprocessing import standardscaler
ss=standardscaler()
X = ss.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

classifier = SupervisedDBNClassification(hidden_layers_structure =      
[256, 256],
learning_rate_rbm=0.05,
learning_rate=0.1,
n_epochs_rbm=10,
n_iter_backprop=100,
batch_size=32,
activation_function='relu',
dropout_p=0.2)

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
