import soundfile as sf
import numpy as np
import os
import time

np.seterr(divide='ignore', invalid='ignore')
import scipy
import scipy.signal
import scipy.fftpack
import pandas as pd
import config

def STFT(x, fr, fs, Hop, h):
	t = np.arange(0, np.ceil(len(x) / float(Hop)) * Hop, Hop)
	N = int(fs / float(fr))
	window_size = len(h)
	f = fs * np.linspace(0, 0.5, int(np.round(N / 2)), endpoint=True)
	Lh = int(np.floor(float(window_size - 1) / 2))
	tfr = np.zeros((int(N), len(t)), dtype=np.float32)

	for icol in range(0, len(t)):
		ti = int(t[icol])
		tau = np.arange(int(-min([round(N / 2.0) - 1, Lh, ti - 1])), \
						int(min([round(N / 2.0) - 1, Lh, len(x) - ti])))
		indices = np.mod(N + tau, N) + 1
		tfr[indices - 1, icol] = x[ti + tau - 1] * h[Lh + tau - 1] \
								 / np.linalg.norm(h[Lh + tau - 1])
	start = time.time()
	tfr = abs(scipy.fftpack.fft(tfr, n=N, axis=0))
	print('fft time:', time.time() - start)
	return tfr, f, t, N


def nonlinear_func(X, g, cutoff):
	cutoff = int(cutoff)
	if g != 0:
		X[X < 0] = 0
		X[:cutoff, :] = 0
		X[-cutoff:, :] = 0
		X = np.power(X, g)
	else:
		X = np.log(X)
		X[:cutoff, :] = 0
		X[-cutoff:, :] = 0
	return X


def Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOct):
	StartFreq = fc
	StopFreq = 1 / tc
	Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
	central_freq = []

	for i in range(0, Nest):
		CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
		if CenFreq < StopFreq:
			central_freq.append(CenFreq)
		else:
			break
	
	'''
	for i in range(len(central_freq)):
		print(i, central_freq[i])
	# print(len(central_freq))
	sys.exit()
	'''

	Nest = len(central_freq)
	freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float32)
	import bisect
	for i in range(1, Nest - 1):
		# l = int(round(central_freq[i - 1] / fr))
		# r = int(round(central_freq[i + 1] / fr) + 1)
		# interval (l,r) (i.e. not including l, r)
		l = bisect.bisect_right(X, central_freq[i-1])
		r = bisect.bisect_left(X, central_freq[i+1])
		# rounding1
		if l >= r - 1:
			freq_band_transformation[i, l] = 1
		else:
			for j in range(l, r):
				if f[j] > central_freq[i - 1] and f[j] <= entral_freq[i]:
					freq_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (
							central_freq[i] - central_freq[i - 1])
				elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
					freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (
							central_freq[i + 1] - central_freq[i])
	tfrL = np.dot(freq_band_transformation, tfr)
	
	
	# print(len(tfrL), len(central_freq))
	# sys.exit()
	return tfrL, central_freq


def Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOct):
	StartFreq = fc
	StopFreq = 1 / tc
	Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
	central_freq = []

	for i in range(0, Nest):
		CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
		if CenFreq < StopFreq:
			central_freq.append(CenFreq)
		else:
			break
	f = 1 / q
	
	# this is basically remapping so that the lenght of cepstrum fit the length of 360 (for spectrum itself, this transform is basicaly x-log)
	# q: from 0 all the way to f_s/f_c (which is the smallest cutoff freq, and therefore the longest "period")
	
	# central_freq, the freq, ranges from [f_c, 1/t_c]
	# hence already reversed here
	
	
	Nest = len(central_freq)
	freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float32)
	for i in range(1, Nest - 1):
		for j in range(int(round(fs / central_freq[i + 1])), int(round(fs / central_freq[i - 1]) + 1)):
			if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
				freq_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (central_freq[i] - central_freq[i - 1])
			elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
				freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])

	tfrL = np.dot(freq_band_transformation, ceps)
	# import sys
	# print(np.nonzero(freq_band_transformation[:, 200:210]))
	# sys.exit()
	return tfrL, central_freq


def CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
	NumofLayer = np.size(g)
	N = int(fs / float(fr))
	[tfr, f, t, N] = STFT(x, fr, fs, Hop, h)
	tfr = np.power(abs(tfr), g[0])
	tfr0 = tfr  # original STFT
	ceps = np.zeros(tfr.shape)

	from config import include_adjusted_exp

	if include_adjusted_exp:
		exp_rate = np.exp(0.0006*f)
	else:
		exp_rate = np.exp(0.0000*f)
	z_trans = np.concatenate([exp_rate, np.flip(exp_rate)], axis = 0)

	# print(f[:10], f[-10:])
	# print(exp_rate[:10], exp_rate[-10:])
	
	# print(z_trans.shape)
	# sys.exit()

	if NumofLayer >= 2:
		for gc in range(1, NumofLayer):
			if np.remainder(gc, 2) == 1:
				tc_idx = round(fs * tc)
				# ceps = np.real(np.fft.fft(tfr, axis=0)) / np.sqrt(N)
				ceps = np.real(np.fft.fft(tfr*np.expand_dims(z_trans, axis = 1), axis=0)) / np.sqrt(N)
				# ceps_2 = np.real(np.fft.fft(tfr, axis=0)) / np.sqrt(N)
				
				ceps = nonlinear_func(ceps, g[gc], tc_idx)
				# ceps_2 = nonlinear_func(ceps_2, g[gc], tc_idx)
			else:
				fc_idx = round(fc / fr)
				tfr = np.real(np.fft.fft(ceps, axis=0)) / np.sqrt(N)
				tfr = nonlinear_func(tfr, g[gc], fc_idx)

	tfr0 = tfr0[:int(round(N / 2)), :]
	tfr = tfr[:int(round(N / 2)), :]
	ceps = ceps[:int(round(N / 2)), :]

	HighFreqIdx = int(round((1 / tc) / fr) + 1)
	f = f[:HighFreqIdx]
	tfr0 = tfr0[:HighFreqIdx, :]
	tfr = tfr[:HighFreqIdx, :]
	HighQuefIdx = int(round(fs / fc) + 1)

	# print(f[:10], f[-10:])
	# print(exp_rate[:HighFreqIdx][:10], exp_rate[:HighFreqIdx][-10:])
	# sys.exit()

	q = np.arange(HighQuefIdx) / float(fs)
	# print("q len", len(q), fs, fc)
	# sys.exit()

	ceps = ceps[:HighQuefIdx, :]
	# ceps_2 = ceps_2[:HighQuefIdx, :]

	tfrL0, central_frequencies = Freq2LogFreqMapping(tfr0, f, fr, fc, tc, NumPerOctave)
	tfrLF, central_frequencies = Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave)
	tfrLQ, central_frequencies = Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave)

	# from dummy_utils import plot_multi_sequences
	# time_index = 200
	# print(np.array(central_frequencies).shape, tfrL0.shape)
	# sys.exit()
	# plot_multi_sequences(central_frequencies[:-1], [tfrL0[:, time_index], tfrLF[:, time_index], tfrLQ[:, time_index]], ["spec", "GCoS", "GC"])
	# plot_multi_sequences(f, [(tfr0**(1/g[0]))[:, time_index], (tfr**(1/g[2]))[:, time_index], (ceps**(1/g[1]))[:, time_index], (tfr0**(1/g[0])*np.expand_dims(np.exp(0.0015*f), axis = 1))[:, time_index]], ["spec", "GCoS", "GC", "spec2"])
	
	# plot_multi_sequences(f, [tfr0[:, time_index], tfr[:, time_index], ceps[:, time_index], ceps_2[:, time_index], (tfr0*np.expand_dims(np.exp(0.00036*f), axis = 1))[:, time_index], ((tfr0**(1/g[0])*np.expand_dims(np.exp(0.0015*f), axis = 1))**g[0])[:, time_index]], ["spec", "GCoS", "GC", "GC2", "spec2", "spec3"])
	
	# from dummy_utils import plot_sequence
	# plot_sequence(list(range(len(z_trans))), z_trans)
	
	# sys.exit()


	return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies


def load_audio(filepath, sr=None, mono=True, dtype='float32'):
	if '.mp3' in filepath:
		from pydub import AudioSegment
		import tempfile
		import os
		mp3 = AudioSegment.from_mp3(filepath)
		_, path = tempfile.mkstemp()
		mp3.export(path, format="wav")
		del mp3
		x, fs = sf.read(path)
		os.remove(path)
	else:
		x, fs = sf.read(filepath)

	if mono == True and len(x.shape) > 1:
		x = np.mean(x, axis=1)
	elif mono == "Left" and len(x.shape) > 1:
		x = x[:, 0]
	elif mono == "Right" and len(x.shape) > 1:
		x = x[:, 1]
		

	if sr:
		x = scipy.signal.resample_poly(x, sr, fs)
		fs = sr
	x = x.astype(dtype)


	# from util import play_sequence
	# play_sequence(x, fs)
	

	return x, fs


def feature_extraction(x, fs, Hop=512, Window=2049, StartFreq=80.0, StopFreq=1000.0, NumPerOct=48):
	fr = 2.0  # frequency resolution
	h = scipy.signal.blackmanharris(Window)  # window size
	g = np.array([0.24, 0.6, 1])  # gamma value

	tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = CFP_filterbank(x, fr, fs, Hop, h, StartFreq, 1 / StopFreq, g, NumPerOct)
	Z = tfrLF * tfrLQ
	time = t / fs
	return Z, time, CenFreq, tfrL0, tfrLF, tfrLQ


def midi2hz(midi):
	return 2 ** ((midi - 69) / 12.0) * 440


def hz2midi(hz):
	return 69 + 12 * np.log2(hz / 440.0)


def get_CenFreq(StartFreq=80, StopFreq=1000, NumPerOct=48):
	Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
	central_freq = []
	for i in range(0, Nest):
		CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
		if CenFreq < StopFreq:
			central_freq.append(CenFreq)
		else:
			break
	return central_freq


def get_time(fs, Hop, end):
	return np.arange(Hop / fs, end, Hop / fs)


def lognorm(x):
	return np.log(1 + x)


def norm(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))

from config import fs, hop
fs = int(fs)
hop = int(hop)

def cfp_process(fpath, ypath=None, csv=False, sr=None, hop=hop, model_type='vocal', mono=True):
	print('CFP process in ' + str(fpath) + ' ... (It may take some times)')
	y, sr = load_audio(fpath, sr=sr, mono=mono)
	if 'vocal' in model_type:
		# 1250
		# 32 2050
		# Z, time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(y, sr, Hop=hop, Window=768, StartFreq=32, StopFreq=2050, NumPerOct=60)
		Z, time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(y, sr, Hop=hop, Window=int(768*fs/8000), StartFreq=32, StopFreq=2050, NumPerOct=60)
	if 'melody' in model_type:
		# Z, time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(y, sr, Hop=hop, Window=768, StartFreq=20.0, StopFreq=2048.0, NumPerOct=60)
		raise NotImplementedError
		
	tfrL0 = norm(lognorm(tfrL0))[np.newaxis, :, :]
	tfrLF = norm(lognorm(tfrLF))[np.newaxis, :, :]
	tfrLQ = norm(lognorm(tfrLQ))[np.newaxis, :, :]
	W = np.concatenate((tfrL0, tfrLF, tfrLQ), axis=0)
	print('Done!')
	print('Data shape: ' + str(W.shape))
	if ypath:
		if csv:
			ycsv = pd.read_csv(ypath, names=["time", "freq"])
			gt0 = ycsv['time'].values
			gt0 = gt0[1:, np.newaxis]

			gt1 = ycsv['freq'].values
			gt1 = gt1[1:, np.newaxis]
			gt = np.concatenate((gt0, gt1), axis=1)
		else:
			gt = np.loadtxt(ypath)
		return W, gt, CenFreq, time
	else:
		return W, CenFreq, time


if __name__ == '__main__':
	datasets = [config.train_file] + config.test_file
	data_dir = "/home/ken/Downloads/labels_and_waveform/"
	cfp_save_dir = "/home/ken/Downloads/cfp_saved/"
	
	print(datasets)
	
	
	# load VOICED version
	# load INSTRUMENTAL version
	
	
	for dataset_index, item in enumerate(datasets):
		txtpath = item
		f = open(txtpath)
		filelists = f.readlines()
		
		for i, file in enumerate(filelists):

			print(i)
			filename = file.rstrip('\n')
			
			
			if "_vocal_only" in filename:
				wavpath = data_dir + filename.replace('_vocal_only.npy', '.wav')
				mono = "Right"
				original_f0path = data_dir + filename.replace('_vocal_only.npy', '.txt')
				
			elif "_instrumental_only" in filename:
				wavpath = data_dir + filename.replace('_instrumental_only.npy', '.wav')
				original_f0path = data_dir + filename.replace('_instrumental_only.npy', '.txt')
				mono = "Left"
			else:
				wavpath = data_dir + filename.replace('.npy', '.wav')
				mono = True
				
				
			import shutil	
			f0path = data_dir + filename.replace('.npy', '.txt')
			if "_vocal_only" in filename and not os.path.isfile(f0path):
				shutil.copyfile(original_f0path, f0path)
			elif "_instrumental_only" in filename and not os.path.isfile(f0path):
				ref_temp = np.loadtxt(original_f0path)
				ref_time = ref_temp[:, 0]
				empty_ref_freq = np.zeros(len(ref_time))
				np.savetxt(f0path, np.c_[ref_time, empty_ref_freq], fmt = "%.3f")
				
				
			
			magfile = cfp_save_dir + filename
			print(magfile)
				
			
			if not os.path.exists(f0path):
				raise Exception("Not f0 file!! for %s" %(f0path))
			
			

			successfully_loaded = False
			if os.path.exists(magfile):
				try:
					np.load(magfile)
					print("Exist:", filename)
					successfully_loaded = True
				except:
					pass
					
			if successfully_loaded == False:
				W, CenFreq, _ = cfp_process(wavpath, sr=fs, mono=mono)

				np.save(magfile, W)
