import gym
import torch

env=gym.make("FrozenLake-v0")

y=bytearray(env.desc).decode()
print(y)

y=torch.Tensor([ord(x) for x in y])
print(y.reshape((4,4)))

print(env.desc)
