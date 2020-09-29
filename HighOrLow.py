import numpy as np

# Hypothesis Function
def h(x, t):
    out=np.dot(x, t)
    out=np.tanh(out)

    return out 

# Cost Function
def MSE(x, y):
    out=0

    # Batching
    for j in range(y.shape[0]):
        out+=(y[j][0]-x[j][0])**2

    # Get the average difference
    out=out*(1/y.shape[0])
    return out

def MSE_individual(x, y):
    return (y-x)**2

def getAverageCost(costs):
    c=0

    for i in range(len(costs)):
        c+=costs[i]

    c=c/len(costs)
    return c

def run():
    n=0.75134246
    o=0.47736686

    t=np.array([[n]])

    x=int(input("Selected Number: "))*(1/50)
    x=np.array([[x]])

    out=h(x, t)
    
    if out<0.5:
        outb="High"
    elif out>0.5:
        outb="Low"

    print("Output: ", h(x, t), f"({outb})")

def test():
    n=0.75134246
    t=np.array([[n]])
    batches=50

    # Selected Number
    x=np.array([np.arange(0, 100, step=1)*(1/50)])
    x=x.reshape((100,1))

    # Actual selected number
    y=np.random.randint(2, size=(1, 100))

    predictions=h(x, t)

    costs=np.array([])
    for i in range(x.shape[0]):
        costs=np.append(costs, (MSE_individual(x[i][0], y[0][i])))
   
    costs=costs.reshape(costs.shape[0], 1)
    return predictions, y, costs

def new_test():
    n=0.61356113
    t=np.array([[n]])
    batches=50

    # Selected Number
    x=np.tanh(np.array([np.arange(0, 100, step=1)]))
    x=x.reshape((100,1))

    # Actual selected number
    y=np.random.randint(2, size=(1, 100))

    predictions=h(x, t)

    costs=np.array([])
    for i in range(x.shape[0]):
        costs=np.append(costs, (MSE_individual(x[i][0], y[0][i])))

    costs=costs.reshape(costs.shape[0], 1)
    return predictions, y, costs

train=False

if train:
    episodes=1

    # Thetas
    t=np.random.random((1, 1))*0.1
    #t=np.array([[0.47150595]])

    # Data collection
    costs=[]
    averageCosts=[]
    lowestCost=9999
    bestLowestCost=9999
    bestWeight=0

    lowest_cost_data={}
    weight_data=[]
    predictions=np.array([])

    while episodes<10:
        batches=600
        alpha=-0.001

        # Selected Number
        x=np.tanh(np.random.randint(100, size=(batches, 1)))

        # Actual selected number
        y=np.random.randint(2, size=(batches, 1))

        # Running
        for i in range(2000):
            prediction=h(x, t)
            cost=MSE(prediction, y)
            costs.append(cost)

            if lowestCost>cost:
                lowestCost=cost
                lowest_cost_data[episodes]={"Weights": t, "Lowest Cost":cost}
            else:
                print("Stopped improving...")
                lowestCost=999
                break

            if bestLowestCost>lowestCost:
                bestLowestCost=lowestCost
                bestWeight=t

            averageCosts.append(getAverageCost(costs))

            print("Episode: ", episodes, "    Epoch: ",i, "    Cost: ", cost)

            temp=np.zeros(t.shape)

            for j in range(t.shape[0]):
                temp[j][0]=t[j][0]-alpha*cost*x[j][0]

            for j in range(t.shape[0]):
                t[j][0]=temp[j][0]

            weight_data.append(t[0][0])

        episodes+=1

    print(bestLowestCost)
    print(bestWeight)

    import matplotlib.pyplot as plt
    f, axs=plt.subplots(3, 2)

    # Costs
    l1,=axs[0,0].plot(costs)
    l2,=axs[0,0].plot(averageCosts)
    
    l1.set_label("Cost")
    l2.set_label("Average Cost")

    axs[0,0].legend()

    axs[0,0].set_xlabel("Epoch")
    axs[0,0].set_ylabel("Cost Value")
    axs[0,0].set_title("Training")

    # Weights
    l1,=axs[0,1].plot(weight_data)

    axs[0,1].set_xlabel("Epoch")
    axs[0,1].set_ylabel("Weight Value")

    l1.set_label("Weight")
    axs[0,1].legend()
    axs[0,1].set_title("Training")

    # Numbers Given vs Actual Values
    predictions, actual, costs1=test()
    n=np.array([np.arange(0, 600, step=1)])
    n=n.reshape((n.shape[1], 1))

    axs[1,0].set_xlabel("Number")
    axs[1,0].set_ylabel("Predictions/Actual")

    n=np.array([np.arange(0,100, step=1)])
    l1=axs[1,0].scatter(n, predictions)
    l2=axs[1,0].scatter(n, costs1)
    #l2=axs[2].scatter(n, actual)
    
    l1.set_label("Predictions")
    l2.set_label("Cost")
    axs[1,0].legend()
    axs[1,0].set_title("Old Model")

    # New
    predictions, actual, costs2=new_test()
    
    n=np.array([np.arange(0, 600, step=1)])
    n=n.reshape((n.shape[1], 1))

    axs[1,1].set_xlabel("Number")
    axs[1,1].set_ylabel("Predictions/Actual")

    n=np.array([np.arange(0,100, step=1)])
    l1=axs[1,1].scatter(n, predictions)
    l2=axs[1,1].scatter(n, costs2)
    #l2=axs[2].scatter(n, actual)
    
    l1.set_label("Predictions")
    l2.set_label("Cost")
    axs[1,1].legend()
    axs[1,1].set_title("New Model")

    # Cost Analysis
    n=np.arange(0, 1, step=0.01)
    costs2=costs2.reshape(costs2.shape[0])
    costs1=costs1.reshape(costs1.shape[0])

    axs[2,0].hist2d(n, costs1)
    axs[2,0].set_xlabel("Number")
    axs[2,0].set_ylabel("Cost")
    axs[2,0].set_title("Old Model Cost Visualized")

    axs[2,1].hist2d(n, costs2)
    axs[2,1].set_xlabel("Number")
    axs[2,1].set_ylabel("Cost")
    axs[2,1].set_title("New Model Cost Visualized")

    plt.show()
else:
    #test()
    run()
