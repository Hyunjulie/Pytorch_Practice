# Practicing Pytorch 

# Learning from tutorials: yunjey's 'Pytorch-tutorial' github

import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
import tqdm
import argparse 

#Setting the device. use gpu if available, or cpu if it's not available 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper-parameters 
input_size = 28*28 #Input size: 28*28 
hidden_size = 500 #This is up to myself.
num_classes = 10 # How many classes I want to classify the picture into 
num_epochs = 5
batch_size = 100 
learning_rate = 0.001 

#Using MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='Hyunjulie/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='Hyunjulie/data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#Choosing which Net to use! 
parser = argparse.ArgumentParser(description = 'Choose which net you would like to use ')
parser.add_argument("user", metavar='H', type=int, help='1: Simple Fully Connected Net, 2: Logistic Regression', required=True)
args = parser.parse_args()

if args.user == 1: 
	model = SimpleNet(input_size, hidden_size, num_classes).to(device)
elif args.user == 2: 
	model = 


# Fully Connected neural network with two hidden layer 
class SimpleNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNetwork, self).__init__()
		self.fc1 = nn.Linear(input_size, 600)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(600, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out 

class LogisticRegression(nn.Module):
	def __init__(self, input_size, num_classes):
		super(LogisticRegression, self).__init__()
		self.input_size = input_size
		self.num_classes = num_classes

	def forward(self, x):
		out = nn.Linear(self.input_size, self.num_classes)

#loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train the model 
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.reshape(-1, 28*28).to(device)
		labels = labels.to(device)

		#forward pass 
		outputs = model(images)
		loss = criterion(outputs, labels)

		#Initializing the gradient to 0 for every backward pass 
		optimizer.zero_grad()

		#backward and optimize
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0: 
			print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))


#Testing 
with torch.no_grad():
	correct = 0 
	total = 0 
	for images, labels in test_loader:
		images = images.reshape(-1, 28*28).to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	print('Accuracy of the network on the 10000 test images : {} %'.format(100 * correct / total))


# Epoch [5/5], Step [200/600], Loss:0.0212
# Epoch [5/5], Step [300/600], Loss:0.0355
# Epoch [5/5], Step [400/600], Loss:0.0188
# Epoch [5/5], Step [500/600], Loss:0.1654
# Epoch [5/5], Step [600/600], Loss:0.0090
# Accuracy of the network on the 10000 test images : 97.87 %
# [Finished in 81.6s]
