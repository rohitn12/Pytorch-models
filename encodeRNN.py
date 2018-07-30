class EncoderRNN(nn.Module):
    def __init__(self , input_size , hidden_nodes):
        super(EncoderRNN , self).__init__()
        self.hidden_nodes = hidden_nodes
        self.embedding = nn.Embedding(input_size , hidden_nodes)
        self.gru = nn.GRU(hidden_nodes , hidden_nodes)
        
    def forward(self , input , hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        output , hidden = self.gru(output , hidden)
        return output , hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
                      
        