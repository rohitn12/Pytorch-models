class RNN(nn.Module):
    def __init__(self , input_size , hidden_layers, num_layers, num_classes ):
        super(RNN , self).__init__()
        self.hidden_layers = hidden_layers
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size , hidden_layers , num_layers , batch_first = True)
        self.fc = nn.Linear(hidden_layers , num_classes)
        
    def forward(self ,x):
        h0 = torch.zeros(self.num_layers , x.size(0) , self.hidden_layers)
        c0 = torch.zeros(self,num_layers , x.size(0) , self.hidden_layers)
        out , _ = self.lstm (x , (h0 , c0))
        out = self.fc(out[:,-1,:])
        return out