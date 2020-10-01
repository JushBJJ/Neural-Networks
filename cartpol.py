import gym
import numpy as np
import matplotlib.pyplot as plt

class model:
    def __init__(self, actions):
        self.Q=np.zeros((1,actions))

        self.lastActon=0
        self.states=[]

        self.alpha=0.05
        self.discount_factor=0.95
        self.epsilon=0.5
        self.actions=actions

    def step(self, old_state, new_state, action, reward):
        best_next_action=np.argmax(self.Q[new_state])
        target=reward+self.discount_factor*self.Q[new_state][best_next_action]
        delta=target-self.Q[old_state][action]
        self.Q[old_state][action]+=self.alpha*delta
        
        return self.Q[old_state][action]

def addState(Q, state):
    n=0

    if str(state) not in Q.states:
        Q.states.append(str(state))
        if Q.Q[0].all()==0:
            Q.Q[0]=np.ones(Q.actions)
        else:
            Q.Q=np.vstack((Q.Q, np.ones(Q.actions)))
        n=1

    return Q, n

env=gym.make("CartPole-v1")
Q=model(2)
state=env.reset()
env.render()

steps=1

costs=[]
cost=0

states=[]
n_states=0

rewards=[]
xrewards=0

avgRewards=[]
totalReward=0
avgReward=0

avgCosts=[]
totalCost=0
avgCost=0

gammas=[]

episodes=1000
done=False

for i in range(1, episodes):
    xrewards=0

    """if i%100==0 and Q.discount_factor<0.8:
        Q.discount_factor+=0.1
"""
    while done==False:
        Q, n=addState(Q, state)
        n_states+=n


        newState, reward, done, info=env.step(Q.lastAction)

        Q, n=addState(Q, newState)
        n_states+=n

        stateA=Q.states.index(str(state))
        stateB=Q.states.index(str(newState))

        cost+=Q.step(stateA, stateB, Q.lastAction, reward)
        totalCost+=cost
        totalReward+=reward

        steps+=1
        state=newState

        if i%1==0:
            print("Cost: ", cost, "\tEpisode: ", i," Step: ", steps, " Reward: ", reward,"gamma: ", Q.discount_factor)
            env.render()

    rewards.append(xrewards)
    costs.append(cost)

    avgReward=totalReward/i
    avgRewards.append(avgReward)

    avgCost=totalCost/i
    avgCosts.append(avgCost)

    states.append(n_states)
    gammas.append(Q.discount_factor)


    cost=0
    done=False
    state=env.reset()

f, axs=plt.subplots(2,2)

#l,=axs[0,0].plot(costs)
l1,=axs[0,0].plot(avgCosts)

axs[0,0].set_xlabel("Episodes")
axs[0,0].set_ylabel("Cost Value")

#l.set_label("Cost")
l1.set_label("Average Costs")

axs[0,0].legend()

l,=axs[0,1].plot(states)
axs[0,1].set_xlabel("Steps")
axs[0,1].set_ylabel("States Discovered")

l.set_label("States")
axs[0,1].legend()

#l,=axs[1,0].plot(rewards)
l1,=axs[1,0].plot(avgRewards)

axs[1,0].set_xlabel("Episodes")
axs[1,0].set_ylabel("rewards")

#l.set_label("Rewards")
l1.set_label("avgRewards")

axs[1,0].legend()

l,=axs[1,1].plot(gammas)

axs[1,1].set_xlabel("Episodes")
axs[1,1].set_ylabel("Gamma Value")
l.set_label("Gamma")
axs[1,1].legend()

plt.show()
