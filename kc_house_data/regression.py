import numpy as np
import matplotlib.pyplot as plt
import csv

# Regression Model
class model:
    def __init__(self, input_features, alpha):
        self.alpha=alpha
        self.input_features=input_features
        self.weights=np.random.random((self.input_features, 1))

        self.cost_history=[]
        self.weight_history=self.weights

    def forward(self, x):
        return np.dot(x, self.weights)
    
    def gradientDescent(self, x, y):
        m=y.shape[0]
        errors=self.forward(x)-y
        hx=np.dot(x.transpose(), errors)
        weight_change=hx*(self.alpha/m)

        self.weights=self.weights-weight_change
        return

    def MSE(self, x, y):
        m=y.shape[0]
        h=self.forward(x)-y # Error of h_theta(x) and y
        
        h_sq=np.square(h) # Squared
        h_s=np.sum(h_sq) # Summed

        cost=h_s/(2*m)
        return cost
    
    def train(self, x, y, epochs):
        for epoch in range(epochs):
            cost=self.MSE(x, y)
            self.cost_history.append(cost)
            self.gradientDescent(x, y)
            self.weight_history=np.append(self.weight_history, self.weights, axis=1)

            print("Epoch: ", epoch, "\tCost: ", cost)

        return self.weight_history, self.cost_history

def normalize(x, y, epoch):
    # Normalize x and y
    for i in range(epoch):
        xcol_maxs=x.max(axis=0)
        xcol_mins=x.min(axis=0)

        ycol_maxs=y.max(axis=0)
        ycol_mins=y.min(axis=0)

        sx=xcol_maxs-xcol_mins
        sy=ycol_maxs-ycol_mins

        ux=x.mean(axis=0)
        uy=y.mean(axis=0)

        x=x-sx
        y=y-sy

        x=np.divide(x, ux)
        y=np.divide(y, uy)
    
    return x,y

houses=[]
categories=None

# Process the Data
with open("kc_house_data.csv", "r") as f:
    reader=csv.DictReader(f, delimiter=",")
    line=0

    for row in reader:
        categories=row.keys()
        houses.append(row)

        line+=1

# Convert to list
categories=list(categories)

# Remove ID and Time
categories.pop(0)
categories.pop(0)

# Initialize x and y features
x=np.array([[]])
y=np.array([])

# Construct x and y features
housex=0

for house in houses:
    nx=np.array([])
    for category in categories:
        nx=np.append(nx, float(house[category]))
   
    if x.size!=0:
        nx=np.array([nx])
        x=np.concatenate((x, nx))
    else:
        x=np.array([np.append(x, nx)])
    
    y=np.append(y, float(house["price"]))

    if housex%1000==0:
        print("House: ", housex,"/",len(houses))

    housex+=1

# Make y into a vector
y=np.array([y])
y=y.transpose()

# Normalize x and y, 2 times
x, y=normalize(x, y, 2)

# Model
regression=model(x.shape[1], alpha=0.01)

# Train
weight_history, j_hist=regression.train(x, y, 1000)

h=regression.forward(x)

# Would be cool if someone could figure out how to unNormalize x and y

# Plot Results
# Predicted and Actual

f, a=plt.subplots(2, 2)
a[0,0].scatter(np.arange(0, y.shape[0]), y)
a[0,0].scatter(np.arange(0, y.shape[0]), h, marker="x")

a[0,0].set_xlabel("Sets")
a[0,0].set_ylabel("Outputs")
a[0,0].legend(["Actual Output","Predicted Output"])

a[0,0].title.set_text("Predictions and Actuals")

# Cost
a[0,1].plot(j_hist)

a[0,1].set_xlabel("Epoch")
a[0,1].set_ylabel("Cost")


a[0,1].legend(["Cost Value"])
a[0,1].title.set_text("Cost over time")

# Weights
for weight_h in range(weight_history.shape[0]):
    a[1,0].plot(weight_history[weight_h])

a[1,0].title.set_text("Weights")
a[1,0].set_xlabel("Epoch")
a[1,0].set_ylabel("Weight Value")

# Show plot
plt.show()
