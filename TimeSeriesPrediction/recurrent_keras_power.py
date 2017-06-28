import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import scipy.io as spio
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
np.random.seed(1234)

def read_mat(path_to_dataset,
             sequence_length=60,
             ratio=1.0):

    max_values = int(round(ratio * 1209600))

    matdata = spio.loadmat(path_to_dataset)
    outer_bin = matdata['Data']
    inner_bin = outer_bin['PQ'][0,0]

    print("How many values imported? ", len(inner_bin[:,0]))

    P_L1N = np.array(inner_bin[0:max_values,0])
    P_L2N = np.array(inner_bin[0:max_values,2])
    P_L3N = np.array(inner_bin[0:max_values,4])

    P_L123N = P_L1N #+ P_L2N + P_L3N

    print("Data loaded from mat. Formatting....")

    result = []

    for index in range(len(P_L123N) - sequence_length):
        result.append(P_L123N[index: index + sequence_length])

    result = np.array(result)

    result_mean = result.mean()
    result -= result_mean
    print("Shift : ", result_mean)
    print("Data  : ", result.shape)
    print(result.shape[0])

    row = int(round(0.9 * result.shape[0]))
    train = result[:row, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[row:, :-1]
    y_test = result[row:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [X_train, y_train, X_test, y_test]



def data_power_consumption(path_to_dataset,
                           sequence_length=50,
                            ratio=1.0):

    max_values = ratio * 2049280

    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=";")
        power = []
        nb_of_values = 0
        for line in data:
            try:
                power.append(float(line[2]))
                nb_of_values += 1
            except ValueError:
                pass
            # 2049280.0 is the total number of valid values, i.e. ratio = 1.0
            if nb_of_values >= max_values:
                break

    print("Data loaded from csv. Formatting...")

    result = []
    for index in range(len(power) - sequence_length):
        result.append(power[index: index + sequence_length])
    

    result = np.array(result)  # shape (2049230, 50)

    result_mean = result.mean()
    result -= result_mean
    print("Shift : ", result_mean)
    print("Data  : ", result.shape)
    print(result.shape[0])

    row = int(round(0.9 * result.shape[0]))
    train = result[:row, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[row:, :-1]
    y_test = result[row:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [X_train, y_train, X_test, y_test]


def build_model():
    model = Sequential()
    layers = [1, 60, 100, 1]
    ### Old Version ###
    
    model.add(LSTM(
        layers[1],
        input_shape=(None, layers[0]),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        layers[3]))
    model.add(Activation("linear"))
    
    '''
    model.add(LSTM(layers[1]))
    model.add(Dropout(0.2))
    model.add(LSTM(layers[2]))
    model.add(Dropout(0.2))
    model.add(Dense(layers[3], activation='linear'))
    model.add(Activation("linear"))
    '''

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()
    epochs = 1
    # ratio 0.5
    ratio = 0.5

    sequence_length = 50
    path_to_dataset = 'household_power_consumption.txt'

    if data is None:
        print('Loading data... ')
        X_train, y_train, X_test, y_test = data_power_consumption(
            path_to_dataset, sequence_length, ratio)
    else:
        X_train, y_train, X_test, y_test = data

    print('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model()

    try:
        model.fit(
            X_train, y_train,
            batch_size=512, nb_epoch=epochs, validation_split=0.05)
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:100])
        plt.plot(predicted[:100])
        plt.show()
    except Exception as e:
        print(str(e))
    print('Training duration (s) : ', time.time() - global_start_time)
    
    return model, y_test, predicted


if __name__ == '__main__':
    print("\n\nChoose input file:")
    print("----------------------------------\n")
    print("\t[1] ADRES_Daten_120208\n")
    print("\t[2] household_power_consumption.txt \n")
    n_input = input("Please enter the number: ")

    if(int(n_input) == 1):
        print("Your choice is \"ADRES_Daten_120208\"")
        run_network(data=read_mat(path_to_dataset="../Input/ADRES_Daten_120208.mat", ratio=1.0))
    if(int(n_input) == 2):
        run_network()

    print("Program will be terminated...")
    time.sleep(3)
