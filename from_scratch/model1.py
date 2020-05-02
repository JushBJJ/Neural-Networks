import numpy as np

np.random.seed(1)

class Dense:
    def __init__(self, in_features, out_features):
        # in_features: Size of x
        # out_features: Size of y (next_neuron/output)

        self.weights=np.random.randn(in_features, out_features)*0.1
        self.biases=np.random.randn((out_features))*0.1

    def forward(self, x):
        return np.dot(x, self.weights) + self.biases

Layer1=Dense(4, 10)
Layer2=Dense(10,5)
x=np.random.randn(1,4)

# Show Layer Infos
print(f"""
Inputs: 
{x}

Layer 1 Weights:
{Layer1.weights}

Layer 1 Biases:
{Layer1.biases}

Layer 2 Weights:
{Layer2.weights}

Layer 2 Biases:
{Layer2.biases}
""")

# Forward

y=Layer1.forward(x)
print("Layer 1 Output: \n",y)
y=Layer2.forward(y)

print("Layer 2 Output: \n", y)
