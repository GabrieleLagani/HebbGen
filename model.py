import torch
import torch.nn as nn
import torch.nn.functional as F

from hebb import HebbianConv2d, HebbianConvTranspose2d
import utils

default_hebb_params = {'mode': HebbianConv2d.MODE_SWTA, 'w_nrm': True, 'k': 50, 'act': nn.Identity(), 'alpha': 1.}

def act(x):
	return torch.relu(x) - 0.2 * torch.relu(-x)

def d_act(x):
	return torch.heaviside(x, values=torch.tensor([0.], device=x.device)) + 0.2 * torch.heaviside(-x, values=torch.tensor([0.], device=x.device))

def act_inv(x):
	return torch.relu(x) - (1 / 0.2) * torch.relu(-x)

class GradCatcher(nn.Module):
	def __init__(self, alpha=0.):
		super().__init__()
		self.alpha = alpha
		
		self.weight = None
		self.delta_w = None
	
	def forward(self, x):
		if self.weight is None:
			self.weight = nn.Parameter(torch.zeros_like(x[0]), requires_grad=True)
			self.delta_w = torch.zeros_like(self.weight)
		
		return x + self.weight.unsqueeze(0)
	
	def compute_update(self, x, y, t=None):
		if self.alpha != 0.:
			self.delta_w[:] = (-(y - (t if t is not None else 0.))).sum(dim=0)
	
	def local_update(self):
		self.delta_w[:] = 0.
		self.weight.grad = None


class Net(nn.Module):
	def __init__(self, input_shape=(3, 32, 32), num_classes=10, hidden_size=256, latent_sampling=False, hebb_params=None, deep_supervision=True, neg_update=False, recirculate=False, flip_update=False):
		super().__init__()
		
		self.input_shape = input_shape
		self.num_classes = num_classes
		self.hidden_size = hidden_size
		self.latent_sampling = latent_sampling
		if hebb_params is None: hebb_params = default_hebb_params
		dpath_hebb_params = {}
		for k, p in hebb_params.items():
			if k == 'mode':
				dpath_hebb_params[k] = p[:-2] if p.endswith('_t') else p
			elif k == 'w_nrm':
				dpath_hebb_params[k] = False
			else:
				dpath_hebb_params[k] = p
		dpath_linear_hebb_params = {k: p if k!='act' else nn.Identity() for k, p in dpath_hebb_params.items()}
		linear_hebb_params = {k: p if k!='act' else nn.Identity() for k, p in hebb_params.items()}
		self.hebb_t = hebb_params['mode'].endswith('_t')
		self.deep_supervision = deep_supervision
		self.neg_update = neg_update
		self.recirculate = recirculate
		self.flip_update = flip_update
		
		# Encoder
		
		self.enc_conv1 = HebbianConv2d(self.input_shape[0], 64, 5, 2, **dpath_hebb_params)
		self.enc_bn1 = nn.BatchNorm2d(64, affine=False)
		
		self.enc_conv2 = HebbianConv2d(64, 128, 3, 2, **dpath_hebb_params)
		self.enc_bn2 = nn.BatchNorm2d(128, affine=False)
		
		self.enc_conv3 = HebbianConv2d(128, 256, 3, 2, **dpath_hebb_params)
		self.enc_bn3 = nn.BatchNorm2d(256, affine=False)
		
		hidden_shape = self.get_hidden_shape()
		self.enc_conv4 = HebbianConv2d(256, self.hidden_size, hidden_shape[-1], 1, **dpath_hebb_params)
		self.enc_bn4 = nn.BatchNorm2d(self.hidden_size, affine=False)
		self.gc4 = GradCatcher(alpha=hebb_params['alpha'])
		
		# Latent encoding
		self.enc_mu = HebbianConv2d(self.hidden_size, self.hidden_size, 1, 1, **dpath_linear_hebb_params)
		self.enc_lv = HebbianConv2d(self.hidden_size, self.hidden_size, 1, 1, **dpath_linear_hebb_params)
		
		# Decoder
		self.dec_conv4 = HebbianConvTranspose2d(self.hidden_size, 256, hidden_shape[-1], 1, **hebb_params)
		self.dec_bn4 = nn.BatchNorm2d(256, affine=False)
		
		self.dec_conv3 = HebbianConvTranspose2d(256, 128, 3, 2, **hebb_params)
		self.dec_bn3 = nn.BatchNorm2d(128, affine=False)
		
		self.dec_conv2 = HebbianConvTranspose2d(128, 64, 3, 2, **hebb_params)
		self.dec_bn2 = nn.BatchNorm2d(64, affine=False)
		
		self.dec_conv1 = HebbianConvTranspose2d(64, self.input_shape[0], 5, 2, **linear_hebb_params)
		self.dec_bn1 = nn.BatchNorm2d(self.input_shape[0], affine=False)
		
		# Additional FC classifier
		self.fc = nn.Linear(self.hidden_size, self.num_classes)
	
	def get_hidden_shape(self):
		self.eval()
		with torch.no_grad(): out = self.forward_features(torch.ones([1, *self.input_shape], dtype=torch.float32))[-1].shape[1:]
		return out
	
	def normalize(self, x, bn):
		with torch.no_grad(): _ = bn(x) # Track stats
		return (x - bn.running_mean.reshape(1, x.shape[1], 1, 1)) / (bn.running_var.reshape(1, x.shape[1], 1, 1)**0.5 + bn.eps)
	
	def diff_inverse(self, y, x0, y0, df_list, inv=None):
		if not isinstance(df_list, (list, tuple)):
			df_list = [df_list]
		return y * df_list[0](self._resize_as(x0, y))
		if inv is not None: return inv(y)
		out = self._resize_as(y0, y)
		for p, df in enumerate(df_list):
			#out = out + (1 / df(self._resize_as(x0, y))) * ((y - self._resize_as(y0, y))**(p+1))
			out = out + (df(self._resize_as(x0, y))) * ((y - self._resize_as(y0, y))**(p+1))
		return out
	
	def norm_inverse(self, y, bn):
		return y / (bn.running_var.reshape(1, y.shape[1], 1, 1)**0.5 + bn.eps)
		#return (y * (bn.running_var.reshape(1, y.shape[1], 1, 1)**0.5 + bn.eps))
		#return y + bn.running_mean.reshape(1, y.shape[1], 1, 1)
		#return (y * (bn.running_var.reshape(1, y.shape[1], 1, 1)**0.5 + bn.eps)) + bn.running_mean.reshape(1, y.shape[1], 1, 1)
	
	def inv_weights(self, y, w, stride):
		O, I, H, W = w.shape
		inv_w = w
		#inv_w = torch.pinverse(w.reshape(w.shape[0], -1), rcond=1e-5).reshape(w.shape[3], w.shape[2], w.shape[1], w.shape[0]).permute(3, 2, 1, 0)
		#inv_w = torch.pinverse(w.reshape(w.shape[0], -1), rcond=1e-5).reshape(w.shape[1], w.shape[2], w.shape[3], w.shape[0]).permute(3, 0, 1, 2)
		return torch.conv_transpose2d(y, inv_w, stride=stride)
	
	def forward_features(self, x, out=None):
		x1 = self.enc_conv1(x)
		x1_act = act(x1)
		x1_nrm = self.normalize(x1_act, self.enc_bn1) #self.enc_bn1(x1_act)
		x2 = self.enc_conv2(x1_nrm)
		x2_act = act(x2)
		x2_nrm = self.normalize(x2_act, self.enc_bn2) #self.enc_bn2(x2_act)
		x3 = self.enc_conv3(x2_nrm)
		x3_act = act(x3)
		x3_nrm = self.normalize(x3_act, self.enc_bn3) #self.enc_bn3(x3_act)
		
		if out is None: out = {}
		out['x'] = x
		out['x1'] = x1
		out['x1_act'] = x1_act
		out['x1_nrm'] = x1_nrm
		out['x2'] = x2
		out['x2_act'] = x2_act
		out['x2_nrm'] = x2_nrm
		out['x3'] = x3
		out['x3_act'] = x3_act
		out['x3_nrm'] = x3_nrm
		
		return out, x3_nrm
	
	def decode_features(self, z, out=None):
		y4 = z
		#y4_nrm = z
		#y4_nrm = act_inv(z * (self.enc_bn4.running_var.reshape(1, -1, 1, 1)**0.5 + self.enc_bn4.eps) + self.enc_bn4.running_mean.reshape(1, -1, 1, 1))
		y4_nrm = (z / (self.enc_bn4.running_var.reshape(1, -1, 1, 1)**0.5 + self.enc_bn4.eps)) * d_act(self._resize_as(out['x4'], z))
		#y4_nrm = y4_nrm + 0.02 * torch.randn_like(y4_nrm)
		
		#y3 = self.dec_conv4(y4_nrm)
		y3 = self.inv_weights(y4_nrm, self.enc_conv4.weight, stride=self.enc_conv4.stride)
		y3_act = act(y3)
		y3_nrm = self.diff_inverse(self.norm_inverse(y3, self.enc_bn3), out['x3'], out['x3_act'], d_act, act_inv)
		#y3_nrm = self.normalize(y3_nrm, self.dec_bn4)
		#y3_nrm = self.normalize(y3_act, self.dec_bn4)
		#y3_nrm = y3_nrm + 0.02 * torch.randn_like(y3_nrm)
		#y2 = self.dec_conv3(y3_nrm)
		y2 = self.inv_weights(y3_nrm, self.enc_conv3.weight, stride=self.enc_conv3.stride)
		y2_act = act(y2)
		y2_nrm = self.diff_inverse(self.norm_inverse(y2, self.enc_bn2), out['x2'], out['x2_act'], d_act, act_inv)
		#y2_nrm = self.normalize(y2_nrm, self.dec_bn3)
		#y2_nrm = self.normalize(y2_act, self.dec_bn3)
		#y2_nrm = y2_nrm + 0.02 * torch.randn_like(y2_nrm)
		#y1 = self.dec_conv2(y2_nrm)
		y1 = self.inv_weights(y2_nrm, self.enc_conv2.weight, stride=self.enc_conv2.stride)
		y1_act = act(y1)
		y1_nrm = self.diff_inverse(self.norm_inverse(y1, self.enc_bn1), out['x1'], out['x1_act'], d_act, act_inv)
		#y1_nrm = self.normalize(y1_nrm, self.dec_bn2)
		#y1_nrm = self.normalize(y1_act, self.dec_bn2)
		#y1_nrm = y1_nrm + 0.02 * torch.randn_like(y1_nrm)
		#y = self.dec_conv1(y1_nrm)
		y = self.inv_weights(y1_nrm, self.enc_conv1.weight, stride=self.enc_conv1.stride)
		y_nrm = self.normalize(y, self.dec_bn1)
		
		if out is None: out = {}
		out['y4'] = y4
		out['y4_nrm'] = y4_nrm
		out['y3'] = y3
		out['y3_act'] = y3_act
		out['y3_nrm'] = y3_nrm
		out['y2'] = y2
		out['y2_act'] = y2_act
		out['y2_nrm'] = y2_nrm
		out['y1'] = y1
		out['y1_act'] = y1_act
		out['y1_nrm'] = y1_nrm
		out['y'] = y
		out['y_nrm'] = y_nrm
		
		return out, y_nrm
	
	def sample(self, x, out=None):
		x4 = self.enc_conv4(x)
		x4 = self.gc4(x4)
		x4_act = act(x4)
		x4_nrm = self.normalize(x4_act, self.enc_bn4)  # self.enc_bn4(x4_act)
		mu, lv = self.enc_mu(x4_nrm), self.enc_lv(x4_nrm)
		z = x4_nrm
		if self.latent_sampling: z = mu + torch.exp(lv * 0.5) * torch.randn_like(mu)
		
		if out is None: out = {}
		out['x4'] = x4
		out['x4_act'] = x4_act
		out['x4_nrm'] = x4_nrm
		out['mu'] = mu
		out['lv'] = lv
		out['z'] = z
		
		return out, z
	
	def classify(self, x, t, out=None):
		clf = self.fc(x.reshape(x.shape[0], -1))
		
		clf_t_pre = None
		clf_t = None
		if t is not None and self.deep_supervision:
			t = utils.dense2onehot(t, self.num_classes)
			clf_t_pre = torch.matmul(((t - clf.softmax(dim=-1)))/clf.shape[0], self.fc.weight)
			clf_t_pre = clf_t_pre.reshape(clf_t_pre.shape[0], clf_t_pre.shape[1], 1, 1)
			clf_t = (clf_t_pre / (self.enc_bn4.running_var.reshape(1, -1, 1, 1)**0.5 + self.enc_bn4.eps)) * d_act(self._resize_as(out['x4'], clf_t_pre))
		
		if out is None: out = {}
		out['clf'] = clf
		out['t'] = t
		out['clf_t_pre'] = clf_t_pre
		out['clf_t'] = clf_t
		
		return out, clf
		
	
	def forward(self, x, t=None):
		# Encoding
		out, x3_nrm = self.forward_features(x)
		
		# Sampling
		out, z = self.sample(x3_nrm, out)
		x4_nrm = out['x4_nrm']
		
		# Decoding
		out, y_nrm = self.decode_features(z, out)
		
		# Classifier
		out, clf = self.classify(x4_nrm, t, out)
		
		# Compute weight updates and return
		if self.training: self.compute_updates(out)
		return out, clf
	
	def _resize_as(self, x, y):
		return F.interpolate(x, y.shape[2:], mode='bilinear', align_corners=False)
	
	def _flip_mix(self, dw1, dw2):
		s = dw1 + dw2
		dw1[:] = s
		dw2[:] = s
	
	def compute_updates(self, features):
		idx = torch.randperm(features['x'].shape[0], device=features['x'].device)
		
		self.eval()
		with torch.no_grad():
			out = features.copy()
			z = features['x4_nrm'] + features['clf_t_pre']
			out['z'] = z
			
			# Decoding
			out, y_nrm = self.decode_features(z, out)
			
			# Encoding
			out, x3_nrm = self.forward_features(self._resize_as(y_nrm, features['x']), out)
		
			# Sampling
			out, z_ = self.sample(x3_nrm, out)
		self.train()
		
		# Encoding layers update
		self.enc_conv1.compute_update(x=features['x'], y=features['x1'], t=self._resize_as(out['y1_nrm'] - features['y1_nrm'], features['x1']) + features['x1'], mult=1)
		if self.neg_update: self.enc_conv1.compute_update(x=features['x'], y=features['x1'], t=(self._resize_as(out['y1_nrm'] - features['y1_nrm'], features['x1']) + features['x1'])[idx], mult=-1)
		self.enc_conv2.compute_update(x=features['x1_nrm'], y=features['x2'], t=self._resize_as(out['y2_nrm'] - features['y2_nrm'], features['x2']) + features['x2'], mult=1)
		if self.neg_update: self.enc_conv2.compute_update(x=features['x1_nrm'], y=features['x2'], t=(self._resize_as(out['y2_nrm'] - features['y2_nrm'], features['x2']) + features['x2'])[idx], mult=1)
		self.enc_conv3.compute_update(x=features['x2_nrm'], y=features['x3'], t=self._resize_as(out['y3_nrm'] - features['y3_nrm'], features['x3']) + features['x3'], mult=1)
		if self.neg_update: self.enc_conv3.compute_update(x=features['x2_nrm'], y=features['x3'], t=(self._resize_as(out['y3_nrm'] - features['y3_nrm'], features['x3']) + features['x3'])[idx], mult=-1)
		self.enc_conv4.compute_update(x=features['x3_nrm'], y=features['x4'], t=self._resize_as(out['y4_nrm'] - features['y4_nrm'], features['x4']) + features['x4'], mult=1)
		if self.neg_update: self.enc_conv4.compute_update(x=features['x3_nrm'], y=features['x4'], t=(self._resize_as(out['y4_nrm'] - features['y4_nrm'], features['x4']) + features['x4'])[idx], mult=-1)
		self.gc4.compute_update(x=features['x4'], y=features['x4'], t=self._resize_as(out['y4_nrm'] - features['y4_nrm'], features['x4']) + features['x4'])
		#self.gc4.compute_update(x=features['x4_nrm'], y=features['x4_nrm'], t=self._resize_as(features['clf_t_pre'], features['x4_nrm']) + features['x4_nrm'])
		
		# Latent variable layers update
		if self.latent_sampling:
			self.enc_mu.compute_update(x=features['x4_nrm'], y=features['mu'], t=features['x4'])
			if self.neg_update: self.enc_mu.compute_update(x=features['x4_nrm'], y=features['mu'], t=features['x4'][idx], mult=-1)
			self.enc_lv.compute_update(x=features['x4_nrm'], y=features['lv'], t=(features['x4'] * (features['z'] - features['mu']) / torch.exp(features['lv']) * 0.5))
			if self.neg_update: self.enc_lv.compute_update(x=features['x4_nrm'], y=features['lv'], t=(features['x4'] * (features['z'] - features['mu']) / torch.exp(features['lv']) * 0.5)[idx], mult=-1)
			kld_loss = torch.mean(-0.5 * torch.sum(1 + features['lv'] - features['mu'] ** 2 - features['lv'].exp(), dim=1), dim=0)
			self.enc_mu.zero_grad()
			self.enc_lv.zero_grad()
			kld_loss.backward(retain_graph=True)
			if self.enc_mu.weight.grad is not None: self.enc_mu.delta_w -= self.enc_mu.weight.grad.clone().detach()
			if self.enc_mu.bias.grad is not None: self.enc_mu.delta_b -= self.enc_mu.bias.grad.clone().detach()
			if self.enc_lv.weight.grad is not None: self.enc_lv.delta_w -= self.enc_lv.weight.grad.clone().detach()
			if self.enc_lv.bias.grad is not None: self.enc_lv.delta_b -= self.enc_lv.bias.grad.clone().detach()
			
		# Decoding layers update
		self.dec_conv4.compute_update(x=features['y4_nrm'], y=features['y3'], t=(self._resize_as(features['x3_nrm'], features['y3']) if self.hebb_t else self._resize_as(features['x4_nrm'], features['z'])))
		#self.dec_conv4.compute_update(x=features['z'], y=features['y3'], t=(self._resize_as(features['x3'], features['y3']) if self.hebb_t else self._resize_as(features['x4'], features['z'])))
		if self.neg_update: self.dec_conv4.compute_update(x=features['z'], y=features['y3'], t=(self._resize_as(features['x3_nrm'], features['y3']) if self.hebb_t else self._resize_as(features['x4_nrm'], features['z']))[idx], mult=-1)
		self.dec_conv3.compute_update(x=features['y3_nrm'], y=features['y2'], t=(self._resize_as(features['x2_nrm'], features['y2']) if self.hebb_t else self._resize_as(features['x3_nrm'], features['y3'])))
		#self.dec_conv3.compute_update(x=features['y3_nrm'], y=features['y2'], t=(self._resize_as(features['x2'], features['y2']) if self.hebb_t else self._resize_as(features['x3'], features['y3'])))
		if self.neg_update: self.dec_conv3.compute_update(x=features['y3_nrm'], y=features['y2'], t=(self._resize_as(features['x2_nrm'], features['y2']) if self.hebb_t else self._resize_as(features['x3_nrm'], features['y3']))[idx], mult=-1)
		self.dec_conv2.compute_update(x=features['y2_nrm'], y=features['y1'], t=(self._resize_as(features['x1_nrm'], features['y1']) if self.hebb_t else self._resize_as(features['x2_nrm'], features['y2'])))
		#self.dec_conv2.compute_update(x=features['y2_nrm'], y=features['y1'], t=(self._resize_as(features['x1'], features['y1']) if self.hebb_t else self._resize_as(features['x2'], features['y2'])))
		if self.neg_update: self.dec_conv2.compute_update(x=features['y2_nrm'], y=features['y1'], t=(self._resize_as(features['x1_nrm'], features['y1']) if self.hebb_t else self._resize_as(features['x2_nrm'], features['y2']))[idx], mult=-1)
		self.dec_conv1.compute_update(x=features['y1_nrm'], y=features['y'], t=(self._resize_as(features['x'], features['y']) if self.hebb_t else self._resize_as(features['x1_nrm'], features['y1'])))
		#self.dec_conv1.compute_update(x=features['y1_nrm'], y=features['y'], t=(self._resize_as(features['x'], features['y']) if self.hebb_t else self._resize_as(features['x1'], features['y1'])))
		if self.neg_update: self.dec_conv1.compute_update(x=features['y1_nrm'], y=features['y'], t=(self._resize_as(features['x'], features['y']) if self.hebb_t else self._resize_as(features['x1_nrm'], features['y1']))[idx], mult=-1)
		
		
		if self.recirculate:
			# Encoding layers update
			self.enc_conv1.compute_update(x=out['x'], y=out['x1'], t=self._resize_as(out['y1'], features['x1']))
			if self.neg_update: self.enc_conv1.compute_update(x=out['x'], y=out['x1'], t=self._resize_as(out['y1'], features['x1'])[idx], mult=-1)
			self.enc_conv2.compute_update(x=out['x1_nrm'], y=out['x2'], t=self._resize_as(out['y2'], features['x2']))
			if self.neg_update: self.enc_conv2.compute_update(x=out['x1_nrm'], y=out['x2'], t=self._resize_as(out['y2'], features['x2'])[idx], mult=-1)
			self.enc_conv3.compute_update(x=out['x2_nrm'], y=out['x3'], t=self._resize_as(out['y3'], features['x3']))
			if self.neg_update: self.enc_conv3.compute_update(x=out['x2_nrm'], y=out['x3'], t=self._resize_as(out['y3'], features['x3'])[idx], mult=-1)
			self.enc_conv4.compute_update(x=out['x3_nrm'], y=out['x4'], t=self._resize_as(features['clf_t'] if features['clf_t'] is not None else features['z'], features['x4']))
			if self.neg_update: self.enc_conv4.compute_update(x=out['x3_nrm'], y=out['x4'], t=(self._resize_as(features['clf_t'] if features['clf_t'] is not None else features['z'], features['x4']) + (features['x4'] if features['clf_t'] is not None else 0.))[idx], mult=-1)
			
			# Decoding layers update
			self.dec_conv4.compute_update(x=out['z'], y=out['y3'], t=((self._resize_as(features['x3'] - out['x3'], out['y3']) + out['y3']) if self.hebb_t else (self._resize_as(features['x4_nrm'] - out['x4_nrm'], out['z']) + out['z'])), mult=1)
			if self.neg_update: self.dec_conv4.compute_update(x=out['z'], y=out['y3'], t=((self._resize_as(features['x3'] - out['x3'], out['y3']) + out['y3']) if self.hebb_t else (self._resize_as(features['x4_nrm'] - out['x4_nrm'], out['z']) + out['z']))[idx], mult=-1)
			self.dec_conv3.compute_update(x=out['y3_nrm'], y=out['y2'], t=((self._resize_as(features['x2'] - out['x2'], out['y2']) + out['y2']) if self.hebb_t else (self._resize_as(features['x3_nrm'] - out['x3_nrm'], out['y3']) + out['y3'])), mult=1)
			if self.neg_update: self.dec_conv3.compute_update(x=out['y3_nrm'], y=out['y2'], t=((self._resize_as(features['x2'] - out['x2'], out['y2']) + out['y2']) if self.hebb_t else (self._resize_as(features['x3_nrm'] - out['x3_nrm'], out['y3']) + out['y3']))[idx], mult=-1)
			self.dec_conv2.compute_update(x=out['y2_nrm'], y=out['y1'], t=((self._resize_as(features['x1'] - out['x1'], out['y1']) + out['y1']) if self.hebb_t else (self._resize_as(features['x2_nrm'] - out['x2_nrm'], out['y2']) + out['y2'])), mult=1)
			if self.neg_update: self.dec_conv2.compute_update(x=out['y2_nrm'], y=out['y1'], t=((self._resize_as(features['x1'] - out['x1'], out['y1']) + out['y1']) if self.hebb_t else (self._resize_as(features['x2_nrm'] - out['x2_nrm'], out['y2']) + out['y2']))[idx], mult=-1)
			self.dec_conv1.compute_update(x=out['y1_nrm'], y=out['y'], t=((self._resize_as(features['x'] - out['x'], out['y']) + out['y']) if self.hebb_t else (self._resize_as(features['x1_nrm'] - out['x1_nrm'], out['y1']) + out['y1'])), mult=1)
			if self.neg_update: self.dec_conv1.compute_update(x=out['y1_nrm'], y=out['y'], t=((self._resize_as(features['x'] - out['x'], out['y']) + out['y']) if self.hebb_t else (self._resize_as(features['x1_nrm'] - out['x1_nrm'], out['y1']) + out['y1']))[idx], mult=-1)
			
		if self.flip_update:
			self._flip_mix(self.enc_conv1.delta_w, self.dec_conv1.delta_w)
			self._flip_mix(self.enc_conv2.delta_w, self.dec_conv2.delta_w)
			self._flip_mix(self.enc_conv3.delta_w, self.dec_conv3.delta_w)
			self._flip_mix(self.enc_conv4.delta_w, self.dec_conv4.delta_w)
			
	def get_param_groups(self):
		return self.parameters()
		def_params = {'params': [], 'param_names': []}
		reverse_branch = {'params': [], 'param_names': [], 'lr': 1e-3, 'weight_decay': 0}
		for n, p in self.named_parameters():
			if 'dec_conv' in n:
				reverse_branch['params'].append(p)
				reverse_branch['param_names'].append(n)
			else:
				def_params['params'].append(p)
				def_params['param_names'].append(n)
		return [def_params, reverse_branch]
	
	def check_grads(self):
		err = (self.gc4.weight.grad - (-self.gc4.delta_w)).abs().mean().item()
		print("Gradient error: {}".format(err))
	