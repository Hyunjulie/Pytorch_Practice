import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyperparameters 
num_epochs = 1
num_classes = 10 
batch_size = 100 
learning_rate = 0.001

#MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='Hyunjulie/data', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root='Hyunjulie/data', train=False, transform=transforms.ToTensor())

#Data loader 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Convolutional Neural Network (2 layers)

class SimpleConvNet(nn.Module):
	def __init__(self, num_classes=10):
		super(SimpleConvNet, self).__init__()

		#Input filter #: 1, output filter #: 16
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), 
			nn.BatchNorm2d(16), 
			nn.ReLU(), 
			nn.MaxPool2d(kernel_size=2, stride=2))

		#Input filter #: 16, output filter #: 32
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), 
			nn.BatchNorm2d(32), 
			nn.ReLU(), 
			nn.MaxPool2d(kernel_size=2, stride=2))

		self.fc = nn.Linear(7*7*32, num_classes)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		return out 

model = SimpleConvNet(num_classes).to(device)

#Loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training 
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)

		#Forward 
		outputs = model(images)
		loss = criterion(outputs, labels)

		#Backward and optimizer 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0: 
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))


#Testing 
model.eval() #Evaluation mode: Batchnorm uses moving mean/variance instead of mini-batch mean/variance
with torch.no_grad():
	correct = 0
	total = 0 
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	print("Test Accuracy of the model on the 10000 test images: {} %".format(100 * correct / total ))


# Epoch [1/1], Step [100/600], Loss: 0.2268
# Epoch [1/1], Step [200/600], Loss: 0.1192
# Epoch [1/1], Step [300/600], Loss: 0.1807
# Epoch [1/1], Step [400/600], Loss: 0.0232
# Epoch [1/1], Step [500/600], Loss: 0.0131
# Epoch [1/1], Step [600/600], Loss: 0.0240
# Test Accuracy of the model on the 10000 test images: 98.5 %
# [Finished in 83.7s]
