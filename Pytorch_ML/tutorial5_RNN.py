#RNN - MNIST 
#From yunjey's pytorch tutorial

import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper-parmeters 
sequence_length = 28 
input_size = 28 
hidden_size = 128
num_layers = 2 
num_classes = 10 
batch_size = 100
num_epochs = 3
learning_rate = 0.01 

train_dataset = torchvision.datasets.MNIST(root='Hyunjulie/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='Hyunjulie/data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Many to one RNN 
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		#Extra steps for RNN -> Setting initial hidden and cell states to zeros
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

		#shape of out => (batch_size, seq_length, hidden_size)
		out, _ = self.lstm(x, (h0, c0)) 

		#Decoding the hidden state of the last time step 
		out = self.fc(out[:, -1, :])
		return out 

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss() #Logistic regression 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.reshape(-1, sequence_length, input_size).to(device)
		labels = labels.to(device)

		outputs = model(images)
		loss = criterion(outputs, labels)
		optimizer.zero_grad() # initializing the gradient every time 
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0: 
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#Testing 
with torch.no_grad(): #Making sure it's not training -> don't need to calculate gradient 
	correct = 0 
	total = 0 
	for images, labels in test_loader: 
		images = images.reshape(-1, sequence_length, input_size).to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print("The Test Accuracy of the model on 10000 test images : {} %".format(100*correct/total))


# Epoch [3/3], Step [200/600], Loss: 0.1204
# Epoch [3/3], Step [300/600], Loss: 0.0516
# Epoch [3/3], Step [400/600], Loss: 0.0702
# Epoch [3/3], Step [500/600], Loss: 0.1157
# Epoch [3/3], Step [600/600], Loss: 0.0420
# The Test Accuracy of the model on 10000 test images : 98.1 %
# [Finished in 379.4s]