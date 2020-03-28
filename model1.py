import torch
import torch.functional as F

from torch import nn
from torch.optim import Adam
from torch.nn import *

# The goal is to output x to the power of 2
# Example: x=5, 5^2=25

# Model Design
#	1 input node
#	2 Hidden nodes
# 	1 output node
#
# Activation: ReLU
# Optimizer: Adam
#

class Model1(nn.Module):
	def __init__(self):
		super(Model1, self).__init__()
		self.input_n=Linear(1, 2)
		self.hidden_n=Linear(2, 1)
		self.activation=[Sigmoid(), ReLU()]
		self.optimizer=Adam(self.parameters(), lr=1e-2)

	def forward(self, x):
		self.y=self.activation[0](self.input_n(x))
		self.y=self.activation[1](self.hidden_n(self.y))
		return self.y		
	
	def loss(self, correct):
		loss_fn=nn.MSELoss(reduction="sum")
		return loss_fn(self.y, correct)
	
	def backward(self, output,  correct):
		self.loss_d=self.loss(correct)
		self.optimizer.zero_grad()
		
		self.loss_d.backward()
		self.optimizer.step()

	def run(self, epoch=5000):
		input(self.input_n.weight)
		input(self.hidden_n.weight)

		for i in range(epoch):

			self.x=torch.randint(100, (50,1), dtype=torch.float)
			self.correct=self.x*self.x

			print(f"Epoch: {i}")
			self.backward(self.forward(self.x), self.correct)
			print(f"Loss: {self.loss_d}")
		
			self.always_zero=True
			for j in self.y:
				if j!=0:
					self.always_zero=False

			if self.loss_d==0.0:
				break	
torch.seed()
net=Model1()

net.run()
torch.save(net.state_dict(), "/home/jush/Neural-Networks/model1_model.txt")
