import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets , transforms
from torch.autograd import Variable

#MNIST dataset

train_dataset = datasets.MNIST(root = './mnist_data' ,
                              train = True,
                              transform = transforms.ToTensor(),
                              download = True)

test_dataset = datasets.MNIST(root = './mnist_data',
                             train = False , 
                             transform = transforms.ToTensor())

#Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                          batch_size = batch_size,
                                          shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset= test_dataset,
                                         batch_size = batch_size , 
                                         shuffle = False)
class Net(nn.Module):
    def __init__(self):
        super(Net , self).__init__()
        self.conv1 = nn.Conv2d(1 , 10 , kernel_size = 5 )
        self.conv2 = nn.Conv2d(10, 20 , kernel_size= 5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320 , 10)
        
    def forward(self , x):
        input_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(input_size , -1) 
        x = self.fc(x)
        return F.log_softmax(x)
        
model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01 )

def train(epoch):
    model.train()
    for batch_idx , (data,target) in enumerate(train_loader):
        data , target = Variable(data) ,  Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output , target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

				
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data , target = Variable(data, volatile = True) , Variable(target)
        output = model(data)
        
        #sum up batch loss
        
        test_loss += criterion(output , target).data[0]
        
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).cuda().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

for epoch in range(1, 10):
    train(epoch)
    test()				