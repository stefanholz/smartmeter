import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import scipy.io as spio
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
np.random.seed(1234)
phase = 0

def read_mat(path_to_dataset,
             sequence_length=50,
             ratio=1.0):

    max_values = int(round(ratio * 1209600))

    matdata = spio.loadmat(path_to_dataset)
    outer_bin = matdata['Data']
    inner_bin = outer_bin['PQ'][0,0]

    print("How many values imported? ", len(inner_bin[:,0]))

    #P_L1N = np.array(inner_bin[0:max_values,0])
    #P_L2N = np.array(inner_bin[0:max_values,2])
    #P_L3N = np.array(inner_bin[0:max_values,4])

    #P_L123N = P_L1N + P_L2N + P_L3N
    P_L123N = np.array(inner_bin[0:max_values,phase])
    #P_L123N = P_L123N[0::60]
    #P_L123N = np.mean(P_L123N.reshape(-1, 60), axis=1)
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

    row = int(round(0.999 * result.shape[0]))
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

    row = int(round((0.9) * result.shape[0]))
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
    layers = [1, 50, 100, 1]
    ### Old Version ###
    
    model.add(LSTM(
        layers[1],
        input_shape=(None, layers[0]),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

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

def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            #print(curr_frame)
            #time.sleep(1)
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, (window_size-1), predicted[-1], axis=0)
        prediction_seqs.append(predicted)
        print("Outer iteration executet --->", i, "/",(int(len(data)/prediction_len)-1))
    return prediction_seqs
	
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to its correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    #plt.show()
    plt.savefig('phase-{0}.svg'.format(phase))

def run_network(model=None, data=None):
    global_start_time = time.time()
    epochs = 1
    # ratio 0.5
    ratio = 1
    #callbacks = [TensorBoard(log_dir='./logs')]

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
        history=model.fit(
            X_train, y_train,
            batch_size=512, nb_epoch=epochs, validation_split=0.05)
        #predicted = model.predict(X_test)
        #predicted = np.reshape(predicted, (predicted.size,))
        predictions = predict_sequences_multiple(model, X_test, 49, 49)
        print("Vorhersagewerte berechnet")
        #plot_history_loss(history)
        plot_results_multiple(predictions, y_test, 49)
    except KeyboardInterrupt:
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    #try:
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.plot(y_test[:100])
        #ax.plot(y_test[:100])
        #plt.plot(predicted[:100])
        #plt.plot(predicted[:100])
        #plt.savefig('plot{0}.svg'.format(phase))
        #plt.show()
    #except Exception as e:
        #print(str(e))
    print('Training duration (s) : ', time.time() - global_start_time)
    
    return model, y_test, predicted

def plot_history_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    # n_input = input("Please enter the number: ")
	
    # For loop for each phase in the .mat-file 0,179,2
    for x in range(54,56,2):
        print("Computation for phase ", x, " is running....")
        phase = x
        run_network(data=read_mat(path_to_dataset="../Input/ADRES_Daten_120208.mat", ratio=1.0))

    print("Program will be terminated...")
    time.sleep(3)
