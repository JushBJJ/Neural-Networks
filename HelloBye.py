#Neural Network that distinguishes between Hello or something else

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import random
import matplotlib.pyplot as plt

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.responses=["Hello", "Bye"]

        self.fc1=nn.Linear(30, 50)
        self.fc2=nn.Linear(50, 30)
        self.fc3=nn.Linear(30, 10)
        self.fc4=nn.Linear(10, len(self.responses))

    def forward(self, x):
        y=F.relu(self.fc1(x))
        y=F.relu(self.fc2(y))
        y=F.relu(self.fc3(y))
        y=torch.tanh(self.fc4(y))

        return y

def text_to_tensor(text):
    t=torch.ones(1, 30, dtype=torch.float) # 30 Input Neurons

    for i in range(0, len(text)):
        t[0][i]=ord(text[i])

    return t

def generate_random_data(size, x, y):
    hello_count=0
    bye_count=0

    for i in range(size):
        new_data=""
        add_hello=random.randint(0,2)

        already_added=False

        for j in range(0, random.randint(0, 29)):
            if len(new_data)==29:
                break

            new_data+=random.choice(string.printable).lower()

            if already_added==False:
                if add_hello==0  and not "hello" in new_data and len(new_data)+len("hello")<29:
                    new_data+="hello"
                elif add_hello==2 and not "bye" in new_data and len(new_data)+len("bye")<29:
                    new_data+="bye"
        
                already_added=True

        if "hello" in new_data:
            y=torch.cat((y, torch.tensor([[1, 0]])), dim=0)
            hello_count+=1
        else:
            y=torch.cat((y, torch.tensor([[0, 1]])), dim=0)
            bye_count+=1        

        x.append(new_data)
        
        if i%10000==0:
            print(f"Generated: {i} new strings\thello: {hello_count}\tbye: {bye_count}")

    return x, y

def new_text(text):
    text=text_to_tensor(text.lower())
    text.cuda()

    return text

torch.set_default_tensor_type("torch.cuda.FloatTensor")

x=[]
y=torch.tensor([])
losses=[]

x, y=generate_random_data(1000, x, y)
new=torch.tensor([])

new.cuda()
y.cuda()

for i in range(len(x)):
    new_tensor=text_to_tensor(x[i])
    new=torch.cat((new, new_tensor), dim=0)

    if i%10000==0:
        print("New: ", i)

x=new

text=new_text("Hello Dude")

test=model()
test.cuda()

opt=optim.SGD(test.parameters(), lr=1e-4)
loss_fn=nn.MSELoss()
scheduler=optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True)

total_loss=0
epochs=50000
batch_size=128

epoch=0

for z in range(epochs):
    permutation=torch.randperm(x.size()[0])

    for i in range(0, x.size()[0], batch_size):
        j=permutation[i:i+batch_size]

        batch_x=x[j]
        batch_y=y[j]

        output=test(batch_x)

        loss=loss_fn(output, batch_y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        scheduler.step(loss)

        total_loss+=loss.item()
        losses.append(loss.item())
        epoch+=1

    if epoch%1000==0:
        print(f"Epoch: {epoch}\tTotal Loss: {total_loss}")

    total_loss=0
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True)

plt.plot(losses)
plt.show()

responses=["Hello", "Bye"]
out=test(text)
print(out)
