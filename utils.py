import os
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.modules.utils import _triple


# Set rng seed
def set_rng_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# Return formatted string with time information
def format_time(seconds):
	seconds = int(seconds)
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	return str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"

# Convert dense-encoded vector to one-hot encoded
def dense2onehot(tensor, n):
	return torch.zeros(tensor.size(0), n, device=tensor.device).scatter_(1, tensor.unsqueeze(1).long(), 1)

# Returns padding size corresponding to padding "same" for a given kernel size
def get_padding_same(kernel_size):
	kernel_size = _triple(kernel_size)
	return [(k - 1) // 2 for k in kernel_size]

# Returns output size after convolution
def get_conv_output_size(input_size, kernel_size, stride=1, padding=0):
	if padding == 'same': padding = get_padding_same(kernel_size)[0]
	return ((input_size + 2*padding - kernel_size) // stride) + 1

def update_param_stats(param_stats, new_stats):
	for n, s in new_stats.items():
		if n + '.max' not in param_stats: param_stats[n + '.max'] = []
		param_stats[n + '.max'].append(torch.max(s.abs()).item())
		if n + '.nrm' not in param_stats: param_stats[n + '.nrm'] = []
		param_stats[n + '.nrm'].append(torch.sum(s ** 2).item())
	return param_stats

def update_param_dist(param_stats, new_stats):
	for n, s in new_stats.items():
		param_stats[n + '.dist'] = s.reshape(-1).tolist()
		param_stats[n + '.nrm_dist'] = (torch.sum(s ** 2, dim=list(range(1, s.ndim))) if s.ndim > 1 else s**2).reshape(-1).tolist()
	return param_stats

# Save data to csv file
def update_csv(results, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, mode='w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		for name, entries in results.items():
			writer.writerow([name + '_step'] + list(entries.keys() if isinstance(entries, dict) else range(1, len(entries) + 1)))
			writer.writerow([name] + list(entries.values() if isinstance(entries, dict) else entries))

# Save a figure showing time series plots in the specified file
def save_plot(data_dicts, path, xlabel='step', ylabel='result'):
	graph = plt.axes(xlabel=xlabel, ylabel=ylabel)
	for key, data_dict in data_dicts.items():
		graph.plot(list(data_dict.keys()) if isinstance(data_dict, dict) else range(1, len(data_dict) + 1),
		           list(data_dict.values()) if isinstance(data_dict, dict) else data_dict,
		           label=str(key))
	graph.grid(True)
	graph.legend()
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig = graph.get_figure()
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)

# Save a grid of figures showing time series plots in the specified file
def save_grid_plot(data_dicts, path, rows=8, cols=8, xlabel='value', ylabel='count'):
	fig, graphs = plt.subplots(rows, cols, figsize=(48, 12))
	i, j = 0, 0
	for key, data_dict in data_dicts.items():
		graph = graphs[i][j]
		graph.plot(list(data_dict.keys()) if isinstance(data_dict, dict) else range(1, len(data_dict) + 1),
		           list(data_dict.values()) if isinstance(data_dict, dict) else data_dict,
		           label=str(key))
		graph.set_xlabel(xlabel)
		graph.set_ylabel(ylabel)
		graph.grid(True)
		graph.legend()
		i += 1
		if i == rows:
			i = 0
			j += 1
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)
	
# Save a grid of figures showing distribution plots in the specified file
def save_grid_dist(data_dicts, path, rows=8, cols=8, bins=10, xlabel='step', ylabel='result'):
	fig, graphs = plt.subplots(rows, cols, figsize=(48, 12), sharex='row')
	i, j = 0, 0
	for key, values in data_dicts.items():
		graph = graphs[i][j]
		graph.hist(list(values), bins=bins, label=str(key), edgecolor='black')
		graph.set_xlabel(xlabel)
		graph.set_ylabel(ylabel)
		graph.grid(True)
		graph.legend()
		i += 1
		if i == rows:
			i = 0
			j += 1
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)

# Save state dictionary file to specified path
def save_dict(state_dict, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(state_dict, path)
	
# Load state dictionary file from specified path
def load_dict(path, device='cpu'):
	return torch.load(path, map_location=device)


