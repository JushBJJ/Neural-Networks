import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import progressbar
import matplotlib.pyplot as plt

from gym.utils import play
from collections import namedtuple

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		# Map, Pos, State
		self.fc1=nn.Bilinear(4,4, 16)
		self.fc2=nn.Linear(16,32)
		self.fc3=nn.Linear(32,16)
		self.fc4=nn.Linear(16,4)
		self.fc5=nn.Linear(4,1)
		
		self.gamma=0.999
		
	def forward(self, info):
		y=torch.tanh(self.fc1(info[0]+info[2], info[1]))
		y=torch.tanh(self.fc2(y))
		y=torch.tanh(self.fc3(y))
		y=torch.tanh(self.fc4(y))
		y=self.fc5(y).reshape((1,4))
	
		return y

class ReplayMemory(object):
	def __init__(self, size):
		self.size=size
		self.memory=[]
		self.pointer=0
	
		self.Transition=namedtuple("Transition",("state", "reward", "action", "next_state", "info"))

	def append(self, *args):
		if len(self.memory)<self.size:
			self.memory.append(None)

		self.memory[self.pointer]=self.Transition(*args)
		self.pointer=(self.pointer+1)%self.size
	
	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)	
	
	def __len__(self):
		return len(self.memory)
	
def optimize_model(model1, model2, memory, optimizer):
	if len(memory)<1:
		return
	
	transitions=memory.sample(1)
	batch=memory.Transition(*zip(*transitions))
	
	non_final_mask=torch.Tensor(tuple(map(lambda s: s is not None, batch.info)))
	non_final_next_states=torch.cat([s for s in batch.info if s is not None])

	pos_batch=torch.cat(batch.state)
	action_batch=torch.cat(batch.action).long()
	reward_batch=torch.cat(batch.reward)

	info_batch=torch.cat(batch.info)

	state_action_values=model1(info_batch).gather(1, action_batch)
	next_state_values=torch.zeros(1)
	next_state_values[non_final_mask.long()-1]=model2(non_final_next_states).max(1)[0].detach()
	
	expected_state_action_values=(next_state_values*model1.gamma)+reward_batch
	loss=F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
	optimizer.zero_grad()
	loss.backward()
		
	for p in model1.parameters():
		p.grad.data.clamp_(-1,1)
	
	optimizer.step()
	return loss.item()

def train(policy_model, test_model, episodes):
	c=0
	epoch=0

	Left=0
	Down=1
	Right=2
	Up=3

	test_model.eval()

	memory=ReplayMemory(200*episodes)
	optimizer=optim.RMSprop(policy_model.parameters())

	reward_p=[]
	loss_p=[]

	for c in range(episodes):
		lossn=0

		pos=torch.ones(4,4)
		pos[0][0]=2
		current_pos=[0,0]

		env_map=bytearray(env.desc).decode()
		env_map=torch.Tensor([ord(x) for x in env_map])
		env_map=env_map.reshape((4,4))
		
		history=torch.zeros(4,4)
		history[0][0]=5
		state=0

		info=torch.cat((pos, env_map, history))
		info=info.reshape((3,4,4))
		
		env.reset()
		reward=0
		while True:
			y=policy_model.forward(info)
			y=y.reshape(1,4)
			_, b=y.max(1)
			direction=b.item()

			if direction==Left and current_pos[0]-1>=0:
				current_pos[0]-=1
				info[0][current_pos[1]][current_pos[0]]=2
				info[2].reshape(1,16)[0][state]=-5			
			elif direction==Down and current_pos[1]+1<env.ncol:
				current_pos[1]+=1
				info[0][current_pos[1]][current_pos[0]]=2
				info[2].reshape(1,16)[0][state]=-5                          
			elif direction==Right and current_pos[0]+1<env.nrow:
				current_pos[0]+=1
				info[0][current_pos[1]][current_pos[0]]=2
				info[2].reshape(1,16)[0][state]=-5                          
			elif direction==Up and current_pos[1]-1>=0:
				current_pos[1]-=1
				info[0][current_pos[1]][current_pos[0]]=2
				info[2].reshape(1,16)[0][state]=-5                          

			last_state=torch.Tensor([state]).unsqueeze(0)
			state, rewardx, done, next_state=env.step(direction)

			if rewardx==0:
				reward-=1
			else:
				reward+=1
	
			info[0]=torch.zeros(4,4)
			info[0].reshape(1,16)[0][state]=2
			info[2].reshape(1,16)[0][state]-=5
	
			state=torch.Tensor([state])
			reward=torch.Tensor([reward])
	
			actions=y.max(1)[1].view(1,1)
			memory.append(state, reward, actions, next_state, info)
			lossn=optimize_model(policy_model, test_model, memory, optimizer)
			epoch+=1
			
			loss_p.append(lossn)
			reward_p.append(reward.item())
			
			state=int(state)	
	
			if done==True:
				print(f"Episode {c} finished\nReward: {reward.item()}\nLoss: {lossn}\tEpoch: {epoch}")
				print(info[2])
				break
		if c%10==0:
			torch.save(policy_model.state_dict(), "model2_model.txt")
	
	plt.plot(reward_p)
	plt.show()
	
def main():
	policy_net=Network()
	target_net=Network()

	try:
		target_net.load_state_dict(torch.load("model2_model.txt"))
		policy_net.load_state_dict(target_net.state_dict())
	except:
		target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()
	train(policy_net, target_net, 5000)


if __name__=="__main__":
	env=gym.make("FrozenLake-v0")
	main()
	env.close()
