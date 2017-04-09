import numpy as np


def magnitude_power(data, begin, end, sampling_rate=1000 / 8.26, sec=121, band_full_ratio=False):
	from numpy import fft
	t = data[sec * begin:sec * end]
	fourier = fft.rfft(t)
	freq = fft.rfftfreq(len(t), 1. / sampling_rate)

	sum_delta = 0
	sum_theta = 0
	sum_alpha = 0
	sum_beta = 0
	sum_gamma = 0
	sum_full = 0

	for i in range(len(freq)):
		# delta band
		if freq[i] < 4:
			magnitude = np.sqrt(fourier[i].real ** 2 + fourier[i].imag ** 2)
			sum_delta += magnitude ** 2

		# theta band
		if 4 <= freq[i] < 8:
			magnitude = np.sqrt(fourier[i].real ** 2 + fourier[i].imag ** 2)
			sum_theta += magnitude ** 2

		# alpha band
		if 8 <= freq[i] < 16:
			magnitude = np.sqrt(fourier[i].real ** 2 + fourier[i].imag ** 2)
			sum_alpha += magnitude ** 2

		# beta band
		if 16 <= freq[i] < 32:
			magnitude = np.sqrt(fourier[i].real ** 2 + fourier[i].imag ** 2)
			sum_beta += magnitude ** 2

		# gamma band
		if 32 <= freq[i]:
			magnitude = np.sqrt(fourier[i].real ** 2 + fourier[i].imag ** 2)
			sum_gamma += magnitude ** 2

		magnitude = np.sqrt(fourier[i].real ** 2 + fourier[i].imag ** 2)
		sum_full += magnitude ** 2

	if band_full_ratio:
		sum_delta /= sum_full
		sum_theta /= sum_full
		sum_alpha /= sum_full
		sum_beta /= sum_full
		sum_gamma /= sum_full

	return sum_delta, sum_theta, sum_alpha, sum_beta, sum_gamma


def root_mean_square(signal):
	signal = signal ** 2
	yrms = np.sqrt(np.mean(signal))
	return yrms


def discrete_wavelet_transform(signal):
	import pywt
	a, b = pywt.dwt(signal, 'rbio1.5')

	maxa = np.max(a)
	mina = np.min(a)
	meana = np.mean(a)
	stda = np.std(a)
	maxb = np.max(b)
	minb = np.min(b)
	meanb = np.mean(b)
	stdb = np.std(b)

	return maxa, mina, meana, stda, maxb, minb, meanb, stdb


def five_number_summary(signal):
	from scipy import stats as st

	xmin = np.min(signal)
	xmax = np.max(signal)
	xmea = np.mean(signal)
	xmed = np.median(signal)
	xmod = st.mode(signal)[0][0]
	xq1, xq3 = np.percentile(signal, [25, 75])
	xiqr = xq3 - xq1
	xstd = np.std(signal)

	return xmin, xmax, xmea, xmed, xmod, xq1, xq3, xiqr, xstd
