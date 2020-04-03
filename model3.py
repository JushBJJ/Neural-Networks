import torch
import matplotlib.pyplot as plt
import gym
import random
import time

def train(Q, env, episodes, visualization=False):
	epilision=0.7
	gamma=0.6
	alpha=0.1

	total_rewards=[]

	for e in range(episodes):
		episode_done=False
		state=env.reset()
		env.render() if visualization is True else None

		while not episode_done:
			if random.uniform(0,1)<epilision:
				action=env.action_space.sample()
			else:
				action=torch.argmax(Q[state]).item()
		
			next_state, reward, episode_done, prob=env.step(action)
			env.render() if visualization is True else None
	
			Q_Value=Q[state, action]
			Q_Max=torch.max(Q[next_state])
			Q[state, action]=(1-alpha)*Q_Value+alpha*(reward+gamma*Q_Max)
			
			state=next_state
		total_rewards.append(reward)		
		print(f"Episode {e} finished.") if e%1000==0 else None
	plt.plot(total_rewards)
	plt.show()

def test(Q, env, episodes):
	for e in range(episodes):
		print("Episode: ", e)
		episode_done=False
		state=env.reset()
		env.render()
			
		while not episode_done:
			action=torch.argmax(Q[state]).item()
			next_state, reward, episode_done, _=env.step(action)
			env.render()
		state=next_state

env=gym.make("Taxi-v3")
Q=torch.zeros(env.observation_space.n, env.action_space.n)

train(Q, env, 50000, visualization=False)
test(Q, env, 10)
