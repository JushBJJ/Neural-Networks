import torch
import os
import torch.functional as F
import matplotlib.pyplot as plt
import progressbar

from torch import nn, tensor
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
    self.input_n=Linear(1, 3)
    self.hidden_n=Linear(3,6)
    self.hidden_n2=Linear(6,6)
    self.hidden_n3=Linear(6,3)
    self.hidden_n4=Linear(3,1)

    self.activation=[Sigmoid(), ReLU()]
    self.optimizer=Adam(self.parameters(), lr=1e-1)
    self.loss_data=[]

    self.next=10000

  def forward(self, x):
    self.y=self.activation[0](self.input_n(x))
    self.y=self.activation[0](self.hidden_n(self.y))
    self.y=self.activation[0](self.hidden_n2(self.y))
    self.y=self.activation[0](self.hidden_n3(self.y))
    self.y=self.activation[1](self.hidden_n4(self.y))

    return self.y		

  def loss(self, correct):
    loss_fn=nn.MSELoss(reduction="sum")
    return loss_fn(self.y, correct)

  def backward(self, output,  correct):
    self.loss_d=self.loss(correct)
    self.optimizer.zero_grad()
    
    self.loss_d.backward()
    self.optimizer.step()

  def train(self, epoch=200000):
    for i in progressbar.progressbar(range(epoch), redirect_stdout=True):

      self.x=torch.randint(100, (50,1), dtype=torch.float)
      self.correct=self.x*self.x
      self.backward(self.forward(self.x), self.correct)
      self.loss_data.append(self.loss_d.data.item())
      
      if i%10000==0:
        print("Loss: ",self.loss_d.data.item())
        torch.save(self.state_dict(), "model1_model.txt")
    
  def run(self, x):
    print("Output: ", self.forward(x))

net=Model1()
net.train()
net.load_state_dict(torch.load("model1_model.txt"))
#net.run()
print(net.state_dict())