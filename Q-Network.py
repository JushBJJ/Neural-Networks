import matplotlib.pyplot as plt
import numpy as np

# Tictactoe
class Game:
    def __init__(self):
        self.actions=9
        self.game=np.chararray((3,3))
        self.game[:]="#"

        self.rows=self.game.shape[0]
        self.cols=self.game.shape[1]

        self.nd_size=self.game.shape[0]*self.game.shape[1]
        self.turn=0

        self.symbols=[b"X", b"O"]

    def checkWin(self):
        win=0.1

        # Check Horizontal
        for j in range(self.game.shape[0]):
            if np.all(self.game[j]==self.symbols[self.turn]):
                win=self.turn
                break

        # Check Vertical
        for i in range(self.game.shape[1]):
            temp=0

            for j in range(self.game.shape[0]):
                if np.all(self.game[j][i]==self.symbols[self.turn]):
                    temp+=1

            if temp==3:
                win=self.turn
                break

        # Check Diagonal (Left to Right)
        sym=self.symbols[self.turn]

        a=self.game[0][0]==sym
        b=self.game[1][1]==sym
        c=self.game[2][2]==sym

        if a and b and c:
            win=self.turn

        # Check Diagonal (Right to Left)
        a=self.game[0][2]==sym
        b=self.game[1][1]==sym
        c=self.game[2][0]==sym

        if a and b and c:
            win=self.turn

        # Check if game is draw
        if win!=self.turn:
            if b"#" in self.game and b"X" in self.game and b"O" in self.game:
                win=0.1
            elif b"#" not in self.game:
                win=-1

        # -1 = Draw
        # 1 = O Wins
        # 0 = X Wins
        # 0.1 = Game still going

        return win

    def doAction(self, action):
        repeated=False

        if action<3 and self.game[0][action]==b"#":
            self.game[0][action]=self.symbols[self.turn]

        elif action>=3 and action<6:
            if self.game[1][action-3]==b"#":
                self.game[1][action-3]=self.symbols[self.turn]
            else:
                repeated=True

        elif action>=6:
            if self.game[2][action-6]==b"#":
                self.game[2][action-6]=self.symbols[self.turn]
            else:
                repeated=True
        else:
            repeated=True

        result=self.checkWin()

        # -1 = Draw
        # 1 = O Wins
        # 0 = X Wins
        # 0.1 = Game still going

        if result==-1:
            reward=0.5
            done=True

        elif result==self.turn:
            reward=1
            done=True

        elif result==0.1:
            reward=-0.1
            done=False
        else:
            reward=-1
            done=True

        if repeated==False:
            if self.turn==0:
                self.turn=1
            else:
                self.turn=0

        state=self.game.reshape(self.nd_size)
        return reward, state, done, repeated

    def reset(self):
        self.game[:]=b"#"
        state=self.game.reshape(self.nd_size)
        self.turn=0
        return state

    def render(self):
        for j in range(self.rows):
            for i in range(self.cols):
                print(self.game[j][i]," ", end="")

            print()

class model:
    def __init__(self, actions):
        self.Q=np.zeros((1,actions))

        self.lastActon=0
        self.states=[]

        self.alpha=0.001
        self.discount_factor=0.9
        self.epsilon=0.1
        self.actions=actions

    def step(self, old_state, new_state, action, reward):
        oldQ=self.Q[old_state][action]
        self.Q[old_state][action]=oldQ+self.alpha*(reward+self.discount_factor*self.Q[new_state, ].argmax()-oldQ)

        return self.Q[old_state][action]

    def policy(self, state):
        A=np.ones(self.actions, dtype=float)*self.epsilon/self.actions
        best_action=np.argmax(self.Q[state])
        A[best_action]+=(1.0-self.epsilon)
        return A

def addState(Q, Q2, state):
    n=0

    if str(state) not in Q.states:
        Q.states.append(str(state))
        if Q.Q[0].all()==0:
            Q.Q[0]=np.ones(Q.actions)
        else:
            Q.Q=np.vstack((Q.Q, np.ones(Q.actions)))
        n=1

    if str(state) not in Q2.states:
        Q2.states.append(str(state))
        if Q2.Q[0].all()==0:
            Q2.Q[0]=np.ones(Q2.actions)
        else:
            Q2.Q=np.vstack((Q2.Q, np.ones(Q2.actions)))

    return Q, Q2, n

env=Game()
Q=model(env.actions)
Q2=model(env.actions)
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

episodes=100000
done=False

for i in range(1, episodes):
    xrewards=0

    if i%15000==0 and Q.discount_factor<0.8:
        Q.discount_factor+=0.1
        Q2.discount_factor+=0.1

    while done==False:
        Q, Q2, n=addState(Q, Q2, state)
        n_states+=n

        probs=Q.policy(Q2.states.index(str(state)))
        Q.lastAction=np.random.choice(np.arange(len(probs)), p=probs)

        reward, newState, done, repeated=env.doAction(Q.lastAction)

        Q, Q2, n=addState(Q, Q2, newState)
        n_states+=n

        stateA=Q.states.index(str(state))
        stateB=Q.states.index(str(newState))

        if repeated==True:
            reward=-1
            cost+=Q.step(stateA, stateB, Q.lastAction, reward)
            totalCost+=cost
            totalReward+=reward
            continue

        cost+=Q.step(stateA, stateB, Q.lastAction, reward)
        totalCost+=cost
        totalReward+=reward

        if done:
            stateA=Q2.states.index(str(state))
            stateB=Q2.states.index(str(newState))
            Q2.step(stateA, stateB, Q2.lastAction, -1)

        else:
            while True:
                prob=Q2.policy(Q2.states.index(str(state)))

                Q2.lastAction=np.random.choice(np.arange(len(prob)), p=prob)
                reward2, newState, done, repeated=env.doAction(Q2.lastAction)

                Q, Q2, n=addState(Q, Q2, state)

                stateA=Q2.states.index(str(state))
                stateB=Q2.states.index(str(newState))

                if repeated==True:
                    cost+=Q2.step(stateA, stateB, Q2.lastAction, -1)
                    continue

                Q2.step(stateA, stateB, Q2.lastAction, reward2)

                if done:
                    stateA=Q.states.index(str(state))
                    stateB=Q.states.index(str(newState))
                    Q.step(stateA, stateB, Q.lastAction, -1)
                
                break

        steps+=1
        state=newState
<<<<<<< HEAD

        if i%1000==0:
            print("Cost: ", cost, "\tEpisode: ", i," Step: ", steps, " Reward: ", reward,"gamma: ", Q.discount_factor)
            env.render()
=======
        #env.render()
        if i%1000==0:
            print("Cost: ", cost, "\tEpisode: ", i," Step: ", steps, " Reward: ", reward,"gamma: ", Q.discount_factor)
>>>>>>> 1fdee49d516bf0f753badefc9ce7c05dc2c1d0c0

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

l,=axs[0,0].plot(costs)
l1,=axs[0,0].plot(avgCosts)

axs[0,0].set_xlabel("Episodes")
axs[0,0].set_ylabel("Cost Value")

l.set_label("Cost")
l1.set_label("Average Costs")

axs[0,0].legend()

l,=axs[0,1].plot(states)
axs[0,1].set_xlabel("Steps")
axs[0,1].set_ylabel("States Discovered")

l.set_label("States")
axs[0,1].legend()

l,=axs[1,0].plot(rewards)
l1,=axs[1,0].plot(avgRewards)

axs[1,0].set_xlabel("Episodes")
axs[1,0].set_ylabel("rewards")

l.set_label("Rewards")
l1.set_label("avgRewards")

axs[1,0].legend()

l,=axs[1,1].plot(gammas)

axs[1,1].set_xlabel("Episodes")
axs[1,1].set_ylabel("Gamma Value")
l.set_label("Gamma")
axs[1,1].legend()

plt.show()
