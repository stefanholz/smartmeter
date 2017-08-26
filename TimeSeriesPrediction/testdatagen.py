import numpy as np

### Function for creating some test data ###
def createTestData(sequence_length=50, ratio=1.0, width=100, length=200):
	
	data = []
	pattern = np.arange(width)
	#print(pattern)
	
	for index in range(round(length/width)):
		data.extend(pattern)

	#data = np.array(data)
	
	result = []
	
	for index in range(len(data) - sequence_length):
		result.append(data[index: index + sequence_length])
		
	result = np.array(result)
	
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