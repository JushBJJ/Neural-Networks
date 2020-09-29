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
            repeat=True

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
            reward=0.8
            done=False

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
        self.states=[0]

        self.alpha=0.01
        self.discount_factor=0.8
        self.actions=actions

    def step(self, old_state, new_state, action, reward):
        oldQ=self.Q[old_state][action]
        self.Q[old_state][action]=oldQ+self.alpha*(reward+self.discount_factor*self.Q[new_state, ].max()-oldQ)

        return self.Q[old_state][action]

    def getAction(self, state):
        if np.random.uniform()<self.discount_factor:
            action=self.Q[state,:].argmax()
        else:
            action=np.random.choice(self.actions)

        return action

def addState(Q, state):
    Q.states.append(str(state))
    Q.Q=np.vstack((Q.Q, np.zeros(Q.actions)))
    return Q

env=Game()
Q=model(env.actions)
state=env.reset()
env.render()

steps=0

costs=[]
cost=0

states=[]

episodes=10000
done=False

for i in range(episodes):
    while not done:
        if str(state) not in Q.states:
            Q=addState(Q, state)
            states.append(len(states)+1)

        action=Q.getAction(Q.states.index(str(state))-1)
        reward, newState, done, repeated=env.doAction(action)

        if repeated:
            continue

        if str(state) not in Q.states:
            Q=addState(Q, state)
            states.append(len(states)+1)

        stateA=Q.states.index(str(state))
        stateB=Q.states.index(str(newState))

        cost+=Q.step(stateA, stateB, action, reward)
        steps+=1

        state=newState
        env.render()
        print("Cost: ", cost, "\tEpisode: ", i,"\tStep: ", steps, "Reward: ", reward)

    costs.append(cost)
    cost=0
    done=False
    state=env.reset()

f, axs=plt.subplots(1,2)

l,=axs[0].plot(costs)
axs[0].set_xlabel("Episodes")
axs[0].set_ylabel("Cost Value")

l.set_label("Cost")
axs[0].legend()

l,=axs[1].plot(states)
axs[1].set_xlabel("Steps")
axs[1].set_ylabel("States Discovered")

l.set_label("States")
axs[1].legend()

plt.show()
