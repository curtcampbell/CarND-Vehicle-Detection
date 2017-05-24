import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

class CarClassifier:

    def __init__(self, presaved_model_file=None):
        if presaved_model_file is None:
            self.scaler = None
            parameters = {'C': [1, 10]}
            svr = svm.LinearSVC()
            # svr = svm.SVC()
            self.svm_classifier = GridSearchCV(svr, parameters)

            # self.svm = svm.SVC(kernel='linear', verbose=False, cache_size=7000)
        else:
            self._load_state(presaved_model_file)

        return

    def fit(self, data, labels):

        self.scaler = StandardScaler().fit(data)
        scaled_data = self.scaler.transform(data)

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_validation, y_train, y_validation = train_test_split(scaled_data, labels, test_size=0.2, random_state=rand_state)

        self.svm_classifier.fit(X_train, y_train)

        # now test on our training data
        predictions = self.svm_classifier.predict(X_validation)
        acc = accuracy_score(predictions, y_validation)

        return acc

    def save_state(self, file_name='car_classifier_model.pkl'):

        output = open(file_name, 'wb')
        pickle.dump(self, output)

        output.close()

    def _load_state(self, file_name):
        # we open the file for reading
        file = open(file_name, 'rb')

        # load the object from the file into var b
        obj = pickle.load(file)
        self.scaler = obj.scaler
        self.svm_classifier = obj.svm_classifier

        file.close()

    def predict(self, X):
        return self.svm_classifier.predict(X)


