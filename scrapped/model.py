import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, TimeDistributed, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import KFold


def get_data(features_name, response_name):
    f_file = open(features_name, 'r')
    r_file = open(response_name, 'r')

    features = []
    for line in f_file.readlines():
        features.append([float(x) for x in line.split()])

    responses = []
    for x in r_file.readlines():
        responses.append(float(x))

    return np.asarray(features, dtype=np.float32), np.asarray(responses, dtype=np.float32)


class Model:
    def __init__(self, feature_data, response_data):
        self.X = feature_data
        self.y = response_data
        self.length = response_data.shape[0]
        self.model = None


    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters=10, kernel_size=3, padding='same', activation='relu', input_shape=(4, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=5, kernel_size=3, padding='same', activation='relu', input_shape=(4, 1)))
        model.add(MaxPooling1D(pool_size=2))
        # model.add(Flatten())
        model.add(GRU(50, input_shape=(3, 1), return_sequences=True))
        model.add(GRU(25, input_shape=(2, 1), return_sequences=True))
        model.add(GRU(10, input_shape=(2, 1)))
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.SGD(learning_rate=0.0000001)
        model.compile(loss='mse', optimizer=opt, metrics=['mse'])
        
        # print(model.summary())
        self.model = model


    def train(self):
        self.model.fit(self.X, self.y, epochs=10, batch_size=64, verbose=10)


    def test(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)


    def get_speeds(self, X_test, file):
        pred = self.model.predict(X_test)
        # file.write(str(pred[0]))
        return pred


    def cross_validate(self):
        mse_per_fold = []
        kfold = KFold(n_splits=5, shuffle=True)

        fold_no = 1
        for train, test in kfold.split(self.X, self.y):
            self.build_model()

            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')

            # Fit data to model
            history = self.model.fit(self.X[train], self.y[train],
                        batch_size=64,
                        epochs=10,
                        verbose=10,
                        validation_split=0.2)

            # Generate generalization metrics
            scores = self.model.evaluate(self.X[test], self.y[test], verbose=5)
            print(f'Score for fold {fold_no}: {self.model.metrics_names[1]} of {scores[1]}')
            mse_per_fold.append(scores[1])

            fold_no += 1

        return mse_per_fold


if __name__ == '__main__':
    features, response = get_data('preds.txt', 'train.txt')
    features = np.reshape(features, (features.shape[0], features.shape[1], -1))
    model = Model(features, response)
    model.build_model()
    model.train()
    f = open('total_pred.txt', 'w')

    preds = np.zeros((20400,))
    for i in range(20400):
        preds[i] = model.get_speeds(np.reshape(features[i], (1, 4, -1)), f)

    print('Max =', np.max(preds))
    print('Min =', np.min(preds))
    print('Mean =', np.mean(preds))

    np.savetxt('total_pred.txt', preds)
    # scores = model.cross_validate()
    # print('5-fold CV MSE:', sum(scores)/5)