"""
Ke Chen knutchen@ucsd.edu

Tone-Octave Network - utils file

This file contains useful common methods

"""
import os
import numpy as np
import torch
import mir_eval
import config

def index2centf(seq, centfreq):
	centfreq[0] = 0
	re = np.zeros(len(seq))
	for i in range(len(seq)):
		for j in range(len(centfreq)):
			if seq[i] < 0.1:
				re[i] = 0
				break
			elif centfreq[j] > seq[i]:
				# re[i] = j
				
				if j > 1:
					if abs(centfreq[j]/seq[i]) <= abs(seq[i]/centfreq[j - 1]):
						re[i] = j
					else:
						re[i] = j - 1
						# print(seq[i], "got", j-1, centfreq[j - 1], "instead of", j, centfreq[j])
				else:
					re[i] = j
				
				break
	return re  


def freq2octave(freq):
	if freq < 1.0 or freq > 2050:
		return config.octave_class
	else:
		return int(np.round(69 + 12 * np.log2(freq/440)) // 12) 

def freq2tone(freq):
	if freq < 1.0 or freq > 2050:
		return config.tone_class
	else:
		return int(np.round(69 + 12 * np.log2(freq/440)) % 12) 

def tofreq(tone, octave):
	if tone >= config.tone_class or octave >= config.octave_class or octave < 2:
		return 0.0
	else:
		return 440 * 2 ** ((12 * octave + tone * 12 / config.tone_class - 69) / 12)


def pos_weight(data, freq_bins):
	frames = data.shape[-1]
	non_vocal = float(len(data[data == 0]))
	vocal = float(data.size - non_vocal)
	z = np.zeros((freq_bins, frames))
	z[1:,:] += (non_vocal / vocal)
	z[0,:] += vocal / non_vocal
	print(non_vocal, vocal)
	return torch.from_numpy(z).float()

def freq2octave(freq):
	if freq < 1.0 or freq > 1990: 
		return 0
	pitch = round(69 + 12 * np.log2(freq / 440))
	return int(pitch // 12)

def compute_roa(pred, gd):
	pred = pred[gd > 0.1]
	gd = gd[gd > 0.1]
	pred = np.array([freq2octave(d) for d in pred])
	gd = np.array([freq2octave(d) for d in gd])
	return np.sum(pred == gd) / len(pred)


def melody_eval(pred, gd):
	ref_time = np.arange(len(gd)) * 0.01
	ref_freq = gd

	est_time = np.arange(len(pred)) * 0.01
	est_freq = pred

	output_eval = mir_eval.melody.evaluate(ref_time,ref_freq,est_time,est_freq)
	VR = output_eval['Voicing Recall']*100.0 
	VFA = output_eval['Voicing False Alarm']*100.0
	RPA = output_eval['Raw Pitch Accuracy']*100.0
	RCA = output_eval['Raw Chroma Accuracy']*100.0
	ROA = compute_roa(est_freq, ref_freq) * 100.0
	OA = output_eval['Overall Accuracy']*100.0
	eval_arr = np.array([VR, VFA, RPA, RCA, ROA, OA])
	return eval_arr

def tonpy_fn(batch):
	dict_key = batch[0].keys()
	output_batch = {}
	for dk in dict_key:
		output_batch[dk] = np.array([d[dk] for d in batch])
	return output_batch
	
# for 010, 0110 etc.
def area_punish(nn_output, area_len = 3):
	assert area_len >= 3
	'''
	product = 1 - nn_output[:, :-area_len+1]
	for index in range(1, area_len-1):
		product = product*nn_output[:, index:-area_len+1+index]
	product = product*(1 - nn_output[:, area_len-1:])
	'''

	product = (1 - nn_output[:, :-area_len+1])*(1 - nn_output[:, area_len-1:])
	
	temp = 1
	for index in range(1, area_len-1):
		temp = temp*(1 - nn_output[:, index:-area_len+1+index])

	product = product*(1 - temp)
	return product


# for 101, 1001 etc.
def reverse_area_punish(nn_output, area_len = 3):
	assert area_len >= 3


	product = nn_output[:, :-area_len+1]*nn_output[:, area_len-1:]
	temp = 1
	for index in range(1, area_len-1):
		temp = temp*nn_output[:, index:-area_len+1+index]

	product = product*(1 - temp)
	return product

	
import sounddevice as sd
def play_sequence(audio_chunk, f_s):
	sd.play(audio_chunk, f_s, blocking = True)



# ys list of y sequences
def plot_multi_sequences(x, ys, y_names, title = "", initial_visibility = True):

	
	import plotly.graph_objects as go

	# https://community.plotly.com/t/hovertemplate-does-not-show-name-property/36139/2
	fig = go.Figure(data = [go.Scatter(x = x, y = ys[i], name = y_names[i], meta = [y_names[i]], hovertemplate = '%{meta}<br>x=%{x}<br>y=%{y}<extra></extra>') for i in range(len(ys))])
	
	
	fig.update_layout(
		title=title,
		xaxis_title="",
		yaxis_title="",
		font=dict(size=25),
		hoverlabel=dict(font_size=25),
		margin={"l":40, "r":40, "t":40, "b":40},
		autosize=True
	)
	
	
	if not initial_visibility:
		fig.update_traces(visible = 'legendonly')
		
	fig.show(config = {'showTips':False})
	



if torch.cuda.is_available():
	device = torch.device("cuda")  
	print("Using cuda")
else:
	device = torch.device("cpu")
	print("Using cpu")

# only dealing with vocal existence
def median_filter(preds, filter_size = 21):
	# import sys
	# print(preds.shape)
	# oddness
	# assert filter_size % 2 == 1
	
	import torch.nn.functional as F
	preds = torch.from_numpy(preds).float().to(device)
	if filter_size % 2 == 1:
		temp = F.pad(preds, (int(filter_size/2), int(filter_size/2)), "constant")
	else:
		temp = F.pad(preds, (int(filter_size/2), int(filter_size/2) - 1), "constant")
	# print(temp.shape, temp.unfold(dimension = -1, size = filter_size, step = 1).shape)
	preds_filtered = torch.median(temp.unfold(dimension = -1, size = filter_size, step = 1), dim = -1).values
	
	assert preds.shape == preds_filtered.shape
	
	preds_on_off = (preds != 0).int()
	preds_filtered_on_off = (preds_filtered != 0).int()
	
	# 0 -> 0, do not change
	# 1 -> 1, do not change
	# 0 -> 1, take the value
	# 1 -> 0, take the value
	# using multiple sizes (one for up and one for down) will cause inconsistency, hence avoid
	should_replace = preds_on_off*(1 - preds_filtered_on_off) + (1 - preds_on_off)*preds_filtered_on_off
	# print("Here")

	# plot_multi_sequences(torch.arange(len(preds)), [preds.cpu().numpy(), ((1 - should_replace)*preds + should_replace*preds_filtered).cpu().numpy()], ["1", "2"])

	return ((1 - should_replace)*preds + should_replace*preds_filtered).cpu().numpy()

if __name__ == "__main__":

	x = torch.randn(2222)
	x = torch.arange(2).repeat(200).numpy()
	print(median_filter(x, filter_size = 20))

	plot_multi_sequences(torch.arange(2222), [x, median_filter(x)], ["1", "2"])
	
