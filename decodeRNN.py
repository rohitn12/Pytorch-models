class DecoderRNN(nn.Module):
    def __init__(self, hidden_size , output_size):
        super(DecoderRNN , self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size , hidden_size)
        self.gru = nn.GRU(hidden_size , hidden_size)
        self.out = nn.Linear(hidden_size , output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self , input , hidden):
        out = self.Embedding(input , hidden).view(1,1,-1)
        out = F.relu(out)
        out , hidden = self.gru(input , hidden)
        out = self.out(out[0])
        out = self.softmax(out)
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
                      
        