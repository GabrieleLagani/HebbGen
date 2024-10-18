import os
import torch
import torch.nn as nn

import utils
from experiment import run


DATASET = 'cifar10'
PRETRAINED_MODEL = None #'assets/pretrained.pt'
WHITEN_LVL = None
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
MOMENTUM = 0.9
WDECAY=5e-4
SCHED_MILESTONES = range(30, 50, 2)
SCHED_GAMMA = 0.5
ELBO_WEIGHT = 0.
HEBB_PARAMS = {'mode': 'grad_t', 'w_nrm': False, 'bias': False, 'act': nn.Identity(), 'k': 50, 'alpha': 1.}
#HEBB_PARAMS = {'mode': 'hpca', 'w_nrm': False, 'bias': True, 'act': nn.Identity(), 'k': 1, 'alpha': 1.}


if __name__ == '__main__':
	utils.set_rng_seed(0)
	run(
		exp_name=os.path.basename(__file__).rsplit('.', 1)[0],
		dataset=DATASET,
		pretrained_model=PRETRAINED_MODEL,
		whiten_lvl=WHITEN_LVL,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
	    lr=LR,
		momentum=MOMENTUM,
		wdecay=WDECAY,
		sched_milestones=SCHED_MILESTONES,
		sched_gamma=SCHED_GAMMA,
		elbo_weight=ELBO_WEIGHT,
		hebb_params=HEBB_PARAMS,
	)

