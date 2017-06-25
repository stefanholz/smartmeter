from tools import read_mat, plot_array
import numpy as np
import time


# ADRES_1_TAG_ratio: 0.07142857142857
# Length of ADRES_Dataset = 1209600

def counter_devices_get_on(ts, hm_hours):
	mean_pre_values = 0
	on_counter = 0
	plt_end = int(round(3600 * hm_hours))
	index_ptr = 4
	peak_counter = 0

	while True:
		matrix_mean = np.array(ts[index_ptr-4:index_ptr])
		mean_pre_values = matrix_mean.mean()
		print(matrix_mean)
		print("Mean value of the matrix slice: ", mean_pre_values)

		if(ts[index_ptr] > mean_pre_values*1.3):
			peak_value = ts[index_ptr]
			peak_counter = 1
			while True:
				if(ts[index_ptr+peak_counter] > peak_value):
					peak_value = ts[index_ptr+peak_counter]
					peak_counter += 1
				else:
					break
			on_counter += 1
			index_ptr = index_ptr + peak_counter + 4
			print(index_ptr)
		else:
			index_ptr += 1
			print index_ptr

		if (index_ptr > (3600*hm_hours)-5):
			break

	print("Amount of devices put on: ", on_counter)


	# print(test)
	#hm_hours = input("How many hours do you wanna plot? ")



	plot_array(array_to_plot=test,plt_start=0, plt_end=3600*hm_hours)


if __name__ == '__main__':
	house_input = input("Choose your houshold (1-30): ")
	w_phase = input("Which phase? (1-6)")
	hm_hours = input("How many hours for evaluation? ")
	test = read_mat(path_to_dataset="../Input/ADRES_Daten_120208.mat", columns=[(house_input*6)+w_phase-7], ratio=1.0)
	# print(test)
	# plot_array(test, plt_end=3600*4)
	counter_devices_get_on(test, hm_hours)
