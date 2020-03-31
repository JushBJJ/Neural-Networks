import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import progressbar

from torch import nn, tensor
from torch.optim import Adam
from torch.nn import Linear, Sigmoid, ReLU, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

  def forward(self, x):
    self.y=self.activation[0](self.input_n(x))
    self.y=self.activation[0](self.hidden_n(self.y))
    self.y=self.activation[0](self.hidden_n2(self.y))
    self.y=self.activation[0](self.hidden_n3(self.y))
    self.y=self.activation[1](self.hidden_n4(self.y))

    return self.y		

def train(x, model, optimizer):
  global loss_data
  model.train()
  correct=x*x

  optimizer.zero_grad()
  output=model(x)
  loss=F.mse_loss(output, correct)
  loss.backward()
  optimizer.step()
  loss_data.append(loss.data.item())
  
  torch.save(model.state_dict(), "model1_model.txt")
    
def run(x, model):
  global test_loss

  model.eval()

  test_loss=0
  
  for i in range(100):
    prediction=model(torch.Tensor([i]))
    test_loss+=F.mse_loss(prediction, torch.Tensor([i*i]))

  test_loss/=100

  print("Average Loss: ", test_loss.item())
  return test_loss

def test(x, model):
  model.eval()
  print("Prediction: ", model(torch.Tensor([x])))

loss_data=[]
net=Model1()
optimizer=Adam(net.parameters(), lr=1)
scheduler=ReduceLROnPlateau(optimizer, verbose=True)

#net.load_state_dict(torch.load("model1_model.txt"))
#test(torch.Tensor([5]), net)

for epoch in progressbar.progressbar(range(5000), redirect_stdout=True):
  x=torch.randint(100, (50,1), dtype=torch.float)
  train(x, net, optimizer)
  test_loss=run(x, net)
  #scheduler.step(test_loss)

#print(net.state_dict())

plt.plot(loss_data)
plt.show()
