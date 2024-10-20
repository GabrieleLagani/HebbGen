import argparse
from time import time
import os
import shutil
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.tensorboard import SummaryWriter

import data
from model import Net
import utils
import params as P


def run(exp_name, dataset='cifar10', pretrained_model=None, whiten_lvl=None, batch_size=32, epochs=20,
        lr=1e-3, momentum=0.9, wdecay=0., sched_milestones=(), sched_gamma=1., elbo_weight=0., hebb_params=None):
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, default=P.DEFAULT_DEVICE, choices=P.AVAILABLE_DEVICES, help="The device you want to use for the experiment.")
	args = parser.parse_args()
	
	device = args.device
	
	trn_set, tst_set, zca = data.get_data(dataset=dataset, root='datasets', batch_size=batch_size, whiten_lvl=whiten_lvl)
	
	model = Net(hebb_params=hebb_params)
	if pretrained_model is not None: model.load_state_dict(utils.load_dict(pretrained_model), strict=False)
	model.to(device=device)
	model.init_inv_weights()
	
	criterion = nn.CrossEntropyLoss()
	def mse(x, y):
		return ((x - y)**2).mean(dim=tuple(range(1, x.ndim)))
	def kld(mu, lv):
		return torch.mean(-0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp(), dim=1), dim=0)
	def elbo(y, x, mu, lv):
		return torch.mean(mse(y, x) + kld(mu, lv))
	optimizer = optim.SGD(model.get_param_groups(), lr=lr, momentum=momentum, weight_decay=wdecay, nesterov=True)
	scheduler = sched.MultiStepLR(optimizer, milestones=sched_milestones, gamma=sched_gamma)
	
	results = {'trn_loss': {}, 'trn_acc': {}, 'trn_mse': {}, 'tst_loss': {}, 'tst_acc': {}, 'tst_mse': {}}
	weight_stats, weight_update_stats, grad_stats = {}, {}, {}
	weight_dist, weight_update_dist, grad_dist = {}, {}, {}
	best_epoch, best_result = None, None
	if os.path.exists('tboard/{}'.format(exp_name)): shutil.rmtree('tboard/{}'.format(exp_name))
	tboard = SummaryWriter('tboard/{}'.format(exp_name))
	for epoch in range(1, epochs + 1):
		t0 = time()
		print("\nEPOCH {}/{} | {}".format(epoch, epochs, exp_name))
		
		# Training phase
		model.train()
		epoch_loss, epoch_hits, epoch_sqerr, count = 0, 0, 0, 0
		weights, weight_updates, grads = {n: copy.deepcopy(p) for n, p in model.named_parameters()}, {}, {}
		print("Training...")
		for inputs, labels in tqdm(trn_set, ncols=80):
			inputs = inputs.to(device)
			labels = labels.to(device)
			
			# Preprocess data if needed
			if zca is not None: inputs = data.whiten(inputs, zca)
			
			# Output computation
			features, outputs = model(inputs, t=labels)
			
			# Model evaluation
			loss = (1 - elbo_weight) * criterion(outputs, labels) + elbo_weight * elbo(features['y'], model._resize_as(features['x'], features['y']), features['mu'], features['lv'])
			if torch.any(torch.isnan(loss)):
				nan_params = [n for n, p in model.named_parameters() if torch.any(torch.isnan(p))]
				raise RuntimeError("Loss is nan. The following params were found to be nan: {}".format(nan_params))
			epoch_loss += loss.sum().item()
			epoch_hits += (torch.max(outputs, dim=1)[1] == labels).int().sum().item()
			epoch_sqerr += mse(features['y'], model._resize_as(features['x'], features['y'])).sum().item()
			count += labels.shape[0]
			
			# Optimization step
			optimizer.zero_grad()
			loss.backward()
			#model.check_grads()
			for m in model.modules():
				if hasattr(m, 'local_update'): m.local_update()
			optimizer.step()
			
			# Track gradients
			for n, p in model.named_parameters():
				if p.grad is None: continue
				grad = p.grad.clone().detach()
				if n not in grads: grads[n] = 0
				grads[n] = grads[n] + grad
		trn_loss, trn_acc, trn_mse = epoch_loss, epoch_hits / count, epoch_sqerr / count
		tboard.add_scalar("Loss/train", trn_loss, epoch)
		tboard.add_scalar("Accuracy/train", trn_acc, epoch)
		tboard.add_scalar("MSE/train", trn_mse, epoch)
		results['trn_loss'][epoch], results['trn_acc'][epoch], results['trn_mse'][epoch] = trn_loss, trn_acc, trn_mse
		print("Train loss: {}, accuracy: {}, mse: {}".format(trn_loss, trn_acc, trn_mse))
		
		# Track weight, weight update, and gradient stats
		for n, p in model.named_parameters():
			if n in weights: weight_updates[n] = p - weights[n]
		weight_stats = utils.update_param_stats(weight_stats, {n: p for n, p in model.named_parameters()})
		weight_dist = utils.update_param_dist(weight_dist, {n: p for n, p in model.named_parameters()})
		for n, s in weight_stats.items(): tboard.add_scalar("Weight/{}".format(n), s[-1], epoch)
		weight_update_stats = utils.update_param_stats(weight_update_stats, weight_updates)
		weight_update_dist = utils.update_param_dist(weight_update_dist, weight_updates)
		for n, s in weight_update_stats.items(): tboard.add_scalar("Delta_W/{}".format(n), s[-1], epoch)
		grad_stats = utils.update_param_stats(grad_stats, grads)
		grad_dist = utils.update_param_dist(grad_dist, grads)
		for n, s in grad_stats.items(): tboard.add_scalar("Grad/{}".format(n), s[-1], epoch)
		
		# Testing phase
		model.eval()
		epoch_loss, epoch_hits, epoch_sqerr, count = 0, 0, 0, 0
		print("Testing...")
		with torch.no_grad():
			for inputs, labels in tqdm(tst_set, ncols=80):
				inputs = inputs.to(device)
				labels = labels.to(device)
				
				# Preprocess data if needed
				if zca is not None: inputs = data.whiten(inputs, zca)
				
				# Output computation
				features, outputs = model(inputs)
				
				# Model evaluation
				loss = (1 - elbo_weight) * criterion(outputs, labels) + elbo_weight * elbo(features['y'], model._resize_as(features['x'], features['y']), features['mu'], features['lv'])
				if torch.any(torch.isnan(loss)): raise RuntimeError("Loss is nan")
				epoch_loss += loss.sum().item()
				epoch_hits += (torch.max(outputs, dim=1)[1] == labels).int().sum().item()
				epoch_sqerr += mse(features['y'], model._resize_as(features['x'], features['y'])).sum().item()
				count += labels.shape[0]
		tst_loss, tst_acc, tst_mse = epoch_loss, epoch_hits / count, epoch_sqerr / count
		tboard.add_scalar("Loss/test", tst_loss, epoch)
		tboard.add_scalar("Accuracy/test", tst_acc, epoch)
		tboard.add_scalar("MSE/test", tst_mse, epoch)
		results['tst_loss'][epoch], results['tst_acc'][epoch], results['tst_mse'][epoch] = tst_loss, tst_acc, tst_mse
		print("Test loss: {}, accuracy: {}, mse: {}".format(tst_loss, tst_acc, tst_mse))
		
		# Keep track of best model
		print("Best model so far at epoch: {}, with result: {}".format(best_epoch, best_result))
		if best_result is None or best_result < tst_acc:
			print("New best model found! Updating best model...")
			best_epoch = epoch
			best_result = tst_acc
			utils.save_dict(copy.deepcopy(model.state_dict()), 'results/{}/best.pt'.format(exp_name))
		
		# Save results
		print("Saving results...")
		utils.update_csv(results, 'results/{}/results.csv'.format(exp_name))
		utils.update_csv(weight_stats, 'results/{}/weight_stats.csv'.format(exp_name))
		utils.update_csv(weight_update_stats, 'results/{}/weight_update_stats.csv'.format(exp_name))
		utils.update_csv(grad_stats, 'results/{}/grad_stats.csv'.format(exp_name))
		utils.update_csv(weight_dist, 'results/{}/weight_dist.csv'.format(exp_name))
		utils.update_csv(weight_dist, 'results/{}/weight_update_dist.csv'.format(exp_name))
		utils.update_csv(grad_dist, 'results/{}/grad_dist.csv'.format(exp_name))
		utils.save_plot({"Train": results['trn_loss'], "Test": results['tst_loss']}, 'results/{}/figures/loss.png'.format(exp_name), xlabel="Epoch", ylabel="Loss")
		utils.save_plot({"Train": results['trn_acc'], "Test": results['tst_acc']}, 'results/{}/figures/accuracy.png'.format(exp_name), xlabel="Epoch", ylabel="Accuracy")
		utils.save_plot({"Train": results['trn_mse'], "Test": results['tst_mse']}, 'results/{}/figures/mse.png'.format(exp_name), xlabel="Epoch", ylabel="Accuracy")
		utils.save_grid_plot(weight_stats, 'results/{}/figures/weight_stats.png'.format(exp_name), rows=2, cols=(len(weight_stats) + 1)//2, ylabel="Weight Value")
		utils.save_grid_plot(weight_update_stats, 'results/{}/figures/weight_update_stats.png'.format(exp_name), rows=2, cols=(len(weight_update_stats) + 1)//2, ylabel="Weight Update")
		utils.save_grid_plot(grad_stats, 'results/{}/figures/grad_stats.png'.format(exp_name), rows=2, cols=(len(grad_stats) + 1)//2, ylabel="Grad. Value")
		utils.save_grid_dist(weight_dist, 'results/{}/figures/weight_dist.png'.format(exp_name), rows=2, cols=(len(weight_dist) + 1)//2, bins=P.DIST_BINS)
		utils.save_grid_dist(weight_update_dist, 'results/{}/figures/weight_update_dist.png'.format(exp_name), rows=2, cols=(len(weight_update_dist) + 1)//2, bins=P.DIST_BINS)
		utils.save_grid_dist(grad_dist, 'results/{}/figures/grad_dist.png'.format(exp_name), rows=2, cols=(len(grad_dist) + 1)//2, bins=P.DIST_BINS)
		tboard.flush()
		utils.save_dict(model.state_dict(), 'results/{}/last.pt'.format(exp_name))
		
		# LR scheduling
		scheduler.step()
		
		t = time() - t0
		print("Epoch duration: {}".format(utils.format_time(t)))
		print("Expected remaining time: {}".format(utils.format_time((epochs - epoch) * t)))
	
	tboard.close()
	
	print("\nFinished!")

