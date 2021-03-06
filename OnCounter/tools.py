import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import csv

def ratio(total_number, hm_hours, hm_values_per_hour):
    return (hm_values_per_hour * hm_hours) / total_number

def plot_array(array_to_plot, plt_start=0, plt_end=100):
    try:
        fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(y_test[:100])
        plt.plot(array_to_plot[plt_start:plt_end])
        plt.show()
    except Exception as e:
        print(str(e))

def plot_mult_arrays(array1, array2, array3, array4, plt_start=0, plt_end=100):
    try:
        # Column 1
        plt.subplot(221)
        plt.plot(array1[plt_start:plt_end])
        plt.title("Phase-1")
        # Column 2
        plt.subplot(222)
        plt.plot(array2[plt_start:plt_end])
        plt.title("Phase-2")
        # Column 3
        plt.subplot(223)
        plt.plot(array3[plt_start:plt_end])
        plt.title("Phase-3")
        # Sum of all columns
        plt.subplot(224)
        plt.plot(array4[plt_start:plt_end])
        plt.title("Sum-of-all-phases")

        plt.show()
    except Exception as e:
        print(str(e))



def read_mat(path_to_dataset,
             columns,
             sequence_length=0,
             ratio=1.0):

    max_values = int(round(ratio * 1209600))

    matdata = spio.loadmat(path_to_dataset)
    outer_bin = matdata['Data']
    inner_bin = outer_bin['PQ'][0,0]

    print("How many values imported? ", (len(inner_bin[:,0]) * ratio))

    output = np.zeros(int(round(ratio * 1209600)))
    for col in columns:
        output = output + np.array(inner_bin[0:max_values, col])
    
    # P_L1N = np.array(inner_bin[0:max_values,3])
    # P_L2N = np.array(inner_bin[0:max_values,4])
    # P_L3N = np.array(inner_bin[0:max_values,5])

    # P_L123N = P_L1N + P_L2N + P_L3N

    print("Data loaded from mat!")
    if sequence_length == 0:
        return output

    result = []

    for index in range(len(output) - sequence_length):
        result.append(output[index: index + sequence_length])

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