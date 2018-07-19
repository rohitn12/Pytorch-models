import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets , transforms
from torch.autograd import Variable


#MNIST dataset
batch_size = 64
train_dataset = datasets.MNIST(root = './mnist_data' ,
                              train = True,
                              transform = transforms.ToTensor(),
                              download = False)

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
										 
class InceptionA(nn.Module):
    def __init__(self,in_channels):
        super(InceptionA , self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels ,16 , kernel_size = 1)
        
        self.branch5x5_1 = nn.Conv2d(in_channels , 16 , kernel_size = 1 )
        self.branch5x5_2 = nn.Conv2d(16 , 24 , kernel_size = 5 , padding = 2)
        
        self.branch3x3_1 = nn.Conv2d(in_channels , 16 , kernel_size = 1)
        self.branch3x3_2 = nn.Conv2d(16 , 24 , kernel_size = 3 ,  padding  = 1)
        self.branch3x3_3 = nn.Conv2d(24 , 24 , kernel_size = 3 , padding = 1)
        
        self.branchpool = nn.Conv2d(in_channels , 24 , kernel_size=1)
    
    def forward(self , x):
        branch1x1 = self.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        
        branchpool = F.avg_pool2d(x , kernel_size = 3 , stride = 1 , padding   = 1)
        branchpool = self.branchpool(branchpool)
        
        outputs = [branch1x1 , branch5x5 , branch3x3 , branchpool]
        outputs = torch.cat(outputs , 1)
        
        return outputs
        
class Net(nn.Module):
    def __init__(self):
        super(Net , self).__init__()
        self.conv1 = nn.Conv2d(1 , 10 , kernel_size = 5 )
        self.conv2 = nn.Conv2d(88  , 20 , kernel_size= 5)
        
        self.incept1 = InceptionA(in_channels= 10)
        self.incept2 = InceptionA(in_channels= 20)
        
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408 , 10)
        
    def forward(self , x):
        input_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x= self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
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