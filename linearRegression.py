import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0] , [2.0], [3.0]  , [4.0] , [5.0]]))
y_data = Variable(torch.Tensor([[3.0] , [6.0] , [9.0] , [12.0] , [15.0]]))

class Model (torch.nn.Module):
    def __init__(self):
        super(Model , self).__init__()
        self.linear = torch.nn.Linear(1,1) #one input and one output 
    def forward(self , x):
        #we give input x and it tries to predict y        
        y_pred = self.linear(x)
        return y_pred
model = Model()

criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters() , lr = 0.01)

for epoch in range(500):
    #Forward Pass
    y_pred = model(x_data)
    
    #compute loss
    
    loss = criterion(y_pred , y_data)
    print("epoch = {} , lossdata = {} " .format(epoch , loss.data[0]))
    
    optimizer.zero_grad() #zerogradients
    loss.backward() #backprop
    optimizer.step() #updata weights

#Testing 

test = Variable(torch.Tensor([[7.0]]))

print("prediction ", model.forward(test).data[0][0])
