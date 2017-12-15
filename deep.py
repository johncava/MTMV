from __future__ import division
from script import *
import numpy as np
import random
import torch
from torch.autograd import Variable

# Definition of Deep Learning Model
class MTMV(torch.nn.Module):
	def __init__(self):
		super(MTMV,self).__init__()
		self.mar = torch.nn.Sequential(
			torch.nn.Linear(64,100),
			torch.nn.Dropout(),
			torch.nn.ReLU(),
			torch.nn.Linear(100,100),
                        torch.nn.Dropout(),
                        torch.nn.ReLU(),
			torch.nn.Linear(100,200),
                        torch.nn.Dropout(),                        
			torch.nn.ReLU()
		)


		self.sha = torch.nn.Sequential(
                        torch.nn.Linear(64,100),
                        torch.nn.Dropout(),
                        torch.nn.ReLU(),
                        torch.nn.Linear(100,100),
                        torch.nn.Dropout(),
                        torch.nn.ReLU(),
                        torch.nn.Linear(100,200),
                        torch.nn.Dropout(),
                        torch.nn.ReLU()
                )

                self.tex = torch.nn.Sequential(
                        torch.nn.Linear(64,100),
                        torch.nn.Dropout(),
                        torch.nn.ReLU(),
                        torch.nn.Linear(100,100),
                        torch.nn.Dropout(),
                        torch.nn.ReLU(),
                        torch.nn.Linear(100,200),
                        torch.nn.Dropout(),
                        torch.nn.ReLU()
                )
		
		self.linear = torch.nn.Sequential(
			torch.nn.Linear(200,200),
			torch.nn.Dropout(),
			torch.nn.ReLU(),
			torch.nn.Linear(200,4),
			torch.nn.Dropout(),
			torch.nn.Sigmoid()
		)
	
	def forward(self, x, y, z):
		x = self.mar(x)
		y = self.sha(y)
		z = self.tex(z)
		add = torch.add(torch.add(x,y),z)
		output = self.linear(add)
		return output 

# Data preparation
data = createDataset()
#random.shuffle(data)
#print data[0]
#print len(data)

#x = data[0]
#x1, x2, x3, y = x
model = MTMV()
loss_fn = torch.nn.MSELoss(size_average = False)
learning_rate = 2e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


k = 5
iterations = 20

for iteration in xrange(iterations):
	sample = random.sample(data,k)
	mar, sha, tex, y  = [], [], [], []
	for item in sample:
		x1, x2, x3, target = item
		mar.append(x1)
		sha.append(x2)
		tex.append(x3)
		y.append(target) 

	mar = np.array(mar)
	sha = np.array(sha)
	tex = np.array(tex)
	y = np.array(y)

	mar = torch.from_numpy(mar)
	mar = mar.float()
	mar = Variable(mar)

	sha = torch.from_numpy(sha)
	sha = sha.float()
	sha = Variable(sha)

	tex = torch.from_numpy(tex)
	tex = tex.float()
	tex = Variable(tex)

	y = torch.from_numpy(y)
	y = y.float()
	y = Variable(y, requires_grad = False)

	y_pred = model(mar, sha, tex)
	#print y_pred
	loss = loss_fn(y_pred, y)
	print loss
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
			
