import numpy as np 
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

class layer_dense:
    def __init__(self, in_features, out_features):
        self.weights=np.random.randn(in_features, out_features) *0.10
        self.biases=np.zeros((1, out_features))

    def forward(self, x):
        self.y=np.dot(x, self.weights) + self.biases

class ReLU:
    def forward(self, x):
        self.output=np.maximum(0, x)

outputs=[]

X,y=spiral_data(100,3)

act=ReLU()
layer1=layer_dense(2,5)
layer2=layer_dense(5,10)

layer1.forward(X)
layer2.forward(layer1.y)

act.forward(layer1.y)
outputs.append(act.output)

act.forward(layer2.y)
outputs.append(act.output)

print(len(outputs))

plt.plot(y)

plt.plot(outputs[0])
plt.plot(outputs[1])

plt.show()