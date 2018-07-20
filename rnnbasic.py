import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigrams = [([test_sentence[i] , test_sentence[i+1]] , test_sentence[i+2]) for i in range(len(test_sentence)-2)]

vocab = set(test_sentence) #getting unique words

word_to_ix = {word : i for i , word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size , embedding_dim , context_size):
        super(NGramLanguageModeler , self).__init__()
        self.embedding = nn.Embedding(vocab_size , embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim , 128)
        self.linear2 = nn.Linear(128 , vocab_size)
        
    def forward(self , inputs):
        embeds = self.embedding(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out , dim =1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab) , EMBEDDING_DIM , CONTEXT_SIZE )
optimizer = optim.SGD(model.parameters() , lr = 0.01)

for epochs in  range(1000):
    total_loss = 0
    for context , target in trigrams:
        # mapping the words in trigrams to indexes
        context_idx = torch.tensor([word_to_ix[w] for w in context] , dtype= torch.long)
        model.zero_grad()
        log_probs = model(context_idx)
        loss = loss_function(log_probs , torch.tensor([word_to_ix[target]] , dtype= torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print(epochs)
print(losses)

#bag of words

EMBEDDING_DIM = 10
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
word_index = {word : i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

def get_max_prob_result(input, ix_to_word):
    return ix_to_word[get_index_of_max(input)]

ix_to_word = {}

for i, word in enumerate(vocab):
    word_index[word] = i
    ix_to_word[i] = word

class CBOW(nn.Module):
    
    def __init__(self , vocab_size , embed_dim ):
        super(CBOW , self).__init__()
        self.embedding = nn.Embedding(vocab_size , embed_dim)
        self.linear1 = nn.Linear(embed_dim , 128)
        self.linear2 = nn.Linear(128 , vocab_size)
        
    def forward(self, inputs):
        embeds = sum(self.embedding(inputs)).view(1,-1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out,dim =1)
        return log_probs

losses = []
model = CBOW(len(vocab) , EMBEDDING_DIM )
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters() , lr = 0.01)

for epochs in range(1000):
    total_loss = 0
    for context , target in data:
        context_vector = make_context_vector(context , word_index)
        model.zero_grad()
        log_probs = model(context_vector)
        loss = loss_function(log_probs ,torch.tensor([word_index[target]] , dtype= torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print(epochs)