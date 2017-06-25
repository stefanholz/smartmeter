import threading
import time
import numpy as np
from tools import read_mat, plot_array, plot_mult_arrays, ratio

exitFlag = 0

class onCounter(threading.Thread):

	def __init__(self, ts, hm_hours, col):
		threading.Thread.__init__(self)
		self.ts = ts
		self.hm_hours = hm_hours
		self.col = col
		print("New thread initialized!")

	def run(self):
		print("Starting thread for column ", self.col)
		start_time = time.time()
		quantity_on = counter_devices_get_on(self.ts, self.hm_hours)
		print("")
		print("    ==== Thread-", self.col, " ====    ")
		print(" Amount of devices turned on: ", quantity_on)
		print("---------------------------------------")
		elapsed_time = time.time() - start_time
		print("Elapsed time for column ", self.col, ": ", elapsed_time)

def counter_devices_get_on(ts, hm_hours):
	mean_pre_values = 0
	on_counter = 0
	plt_end = int(round(3600 * hm_hours))
	index_ptr = 5
	peak_counter = 0
	while True:
		matrix_mean_pre = np.array(ts[index_ptr-5:index_ptr])
		matrix_mean_post = np.array(ts[index_ptr+1:index_ptr+5])

		mean_pre_values = matrix_mean_pre.mean()
		mean_post_values = matrix_mean_post.mean()
		# Uncomment for testing
		#print(matrix_mean)
		#print("Mean value of the matrix slice: ", mean_pre_values)
		#print("Observated value:", ts[index_ptr])
		if(mean_post_values > mean_pre_values*1.25):
			peak_value = ts[index_ptr]
			peak_counter = 1
			while True:
				if(ts[index_ptr+peak_counter] > peak_value):
					peak_value = ts[index_ptr+peak_counter]
					peak_counter += 1
				else:
					break
			on_counter += 1
			# Offset +4 ignoriert die nächsten 4 Werte nach dem Peak
			index_ptr = index_ptr + peak_counter + 4
			# print(index_ptr)
		else:
			index_ptr += 1
			# print(index_ptr)
		if (index_ptr > (3600*hm_hours)-5):
			break
		#time.sleep(100)
	# print("Amount of devices put on: ", on_counter)
	return on_counter
	# print(test)
	#hm_hours = input("How many hours do you wanna plot? ")
	#plot_array(array_to_plot=ts,plt_start=0, plt_end=3600*hm_hours)

if __name__ == '__main__':
	house_input = input("Choose your houshold (1-30): ")
	# w_phase = input("Which phase? (1-6)")
	hm_hours = int(input("How many hours for evaluation? "))
	input1 = read_mat(path_to_dataset="../Input/ADRES_Daten_120208.mat", columns=[(int(house_input)*6)-6], ratio=ratio(1209600.0, hm_hours, 3600.0))
	counter_thread1 = onCounter(input1, hm_hours, 1)
	input2 = read_mat(path_to_dataset="../Input/ADRES_Daten_120208.mat", columns=[(int(house_input)*6)-4], ratio=ratio(1209600.0, hm_hours, 3600.0))
	counter_thread2 = onCounter(input2, hm_hours, 2)
	input3 = read_mat(path_to_dataset="../Input/ADRES_Daten_120208.mat", columns=[(int(house_input)*6)-2], ratio=ratio(1209600.0, hm_hours, 3600.0))
	counter_thread3 = onCounter(input3, hm_hours, 3)
	input_sum = read_mat(path_to_dataset="../Input/ADRES_Daten_120208.mat", columns=[(int(house_input)*6)-6,(int(house_input)*6)-4,(int(house_input)*6)-2], ratio=ratio(1209600.0, hm_hours, 3600.0))
	counter_thread4 = onCounter(input_sum, hm_hours, 4)
	counter_thread1.start()
	counter_thread2.start()
	counter_thread3.start()
	counter_thread4.start()
	# Add threads to thread list
	threads = []
	threads.append(counter_thread1)
	threads.append(counter_thread2)
	threads.append(counter_thread3)
	threads.append(counter_thread4)

	# Wait for all threads to complete
	for t in threads:
		t.join()
	print("All threads successfully terminated!")

	# Plotting diagrams
	plot_mult_arrays(input1, input2, input3, input_sum, plt_end=3600*hm_hours)

	print("Main-Thread finished!")

	# print(test)
	# plot_array(test, plt_end=3600*4)
	# counter_devices_get_on(test, hm_hours)