import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

maximum_letters=20
maximum_words=100
maximum_batches=1

text=[]
vocabulary=[" Cock and ball torture (CBT), penis torture or dick torture is a sexual activity involving application of pain or constriction to the penis or testicles. This may involve directly painful activities, such as genital piercing, wax play, genital spanking, squeezing, ball-busting, genital flogging, urethral play, tickle torture, erotic electrostimulation, kneeing or kicking. The recipient of such activities may receive direct physical pleasure via masochism, or emotional pleasure through erotic humiliation, or knowledge that the play is pleasing to a sadistic dominant. Many of these practices carry significant health risks."]

text_tensor=torch.zeros(maximum_batches, maximum_words, maximum_letters, dtype=torch.float)

def split_text(text_list):
	new_text=[]
	for i in range(len(text_list)):
		new_text.append(text_list[i].split(" "))
	
	text_list=[]
	
	for i in range(len(new_text)):
		for x in new_text[i]:
			text_list.append(x)
	
	return text_list

def line_to_tensor(line):
	placeholder=torch.zeros(maximum_letters)
	for i in range(len(line)):
		placeholder[i]=ord(line[i])

	return placeholder

def text_to_line_ord(textx):
	c=0
	new_tensor=torch.zeros(1,100,20)

	for i in range(len(textx)):
		tensor_line=line_to_tensor(textx[i])
		new_tensor[0][c]=tensor_line
		c+=1
	
	return new_tensor

class network(nn.Module):
	def __init__(self):
		super(network, self).__init__()

		self.input_dim=maximum_letters
		self.vocab=split_text(vocabulary)
		self.vocab_size=len(self.vocab)

		self.recurrent_layers=500
		self.hidden_size=360
		self.hidden=None
		self.output_length=10

		# Layers
		self.rn1=nn.RNN(self.input_dim, self.hidden_size, self.recurrent_layers, batch_first=True)
		self.out1=nn.Linear(self.hidden_size, self.vocab_size)

	def forward(self, x, output_length):
		self.hidden=torch.zeros(self.recurrent_layers, len(x), self.hidden_size)
		self.output_length=output_length

		x=split_text(vocabulary)
		x=text_to_line_ord(x)
		
		y, hidden=self.rn1(x, self.hidden)
		y=torch.tanh(self.out1(y))
		return y

	def chat(self, output):
		print("[Chatbot Start]")
		for i in range(len(output[0])):
			if i>self.output_length:
				break

			print(self.vocab[torch.argmax(output[0][i])], end=" ")				
		print("\n[Chatbot End]")

text.append(input("Input: "))
chatbot=network()
chatbot.train()
y=chatbot(text, 5)
chatbot.chat(y)
