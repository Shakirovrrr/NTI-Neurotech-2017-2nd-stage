import numpy as np
import helping as H


def prepare_trainset(dataset, sec, feature):
	if feature not in featurelist():
		raise TypeError('Feature cannot be ' + '\'' + feature + '\'. ' +
		                'Check the featurelist() for available features.')

	X_train, Y_train = [], []
	state = 0
	count = 0

	for data in dataset:
		a, b = 0, sec
		while b < len(data[0]):
			t = data[:, a:b]
			if feature == 'rms':
				xf = np.asarray([H.root_mean_square(i) for i in t])

			if feature == 'dwt':
				xf = np.asarray([H.discrete_wavelet_transform(i) for i in t])
				xf = np.reshape(xf, -1)

			if feature == 'mag_power':
				xf = np.asarray([H.magnitude_power(i, 0, len(i)) for i in t])
				xf = np.reshape(xf, -1)

			if feature == 'fivenumber':
				xf = np.asarray([H.five_number_summary(i) for i in t])
				xf = np.reshape(xf, -1)

			if feature == 'superfeature':
				xf = np.asarray([list(H.magnitude_power(i, 0, len(i))) +
				                 list(H.discrete_wavelet_transform(i)) +
				                 list(H.five_number_summary(i)) for i in t])
				xf = np.reshape(xf, -1)

			if feature == None:
				xf = np.reshape(t, -1)

			X_train.append(xf)
			Y_train.append(state)

			a += sec
			b += sec

		count += 1
		if 2 <= count <= 3: state = 1
		if 4 <= count <= 5: state = 2

	X_train = np.asarray(X_train)
	Y_train = np.asarray(Y_train)
	assert (len(X_train) == len(Y_train))

	return X_train, Y_train


def prepare_predictset(data, n_times, feature=None, sec=121):
	if feature not in featurelist():
		raise TypeError('Feature cannot be ' + '\'' + feature + '\'. ' +
		                'Check the featurelist() for available features.')

	X_predict = []
	a, b = 0, sec

	for i in range(n_times):
		t = data[:, a:b]
		if feature == 'rms':
			xf = np.asarray([H.root_mean_square(i) for i in t])

		if feature == 'dwt':
			xf = np.asarray([H.discrete_wavelet_transform(i) for i in t])
			xf = np.reshape(xf, -1)

		if feature == 'mag_power':
			xf = np.asarray([H.magnitude_power(i, 0, len(i)) for i in t])
			xf = np.reshape(xf, -1)

		if feature == 'fivenumber':
			xf = np.asarray([H.five_number_summary(i) for i in t])
			xf = np.reshape(xf, -1)

		if feature == 'superfeature':
			xf = np.asarray([list(H.magnitude_power(i, 0, len(i))) +
			                 list(H.discrete_wavelet_transform(i)) +
			                 list(H.five_number_summary(i)) for i in t])
			xf = np.reshape(xf, -1)

		if feature == None:
			xf = np.reshape(t, -1)

		X_predict.append(xf)

		a += sec
		b += sec

	X_predict = np.asarray(X_predict)

	return X_predict


def load_files(files, dir='', delimeter=','):
	dataset = []

	for f in files:
		data = np.genfromtxt(dir + f, delimiter=delimeter)
		asize = data[0].size - 1
		# print(f + ' size: ', asize)
		dataset.append(data[:, :asize])

	return dataset


def load_file(file, dir='', delimeter=',', has_number_line=False):
	data = np.genfromtxt(dir + file, delimiter=delimeter)
	if has_number_line:
		data = data[1:]
	return data


def featurelist():
	return ['rms', 'dwt', 'mag_power', 'fivenumber', 'superfeature', None]
