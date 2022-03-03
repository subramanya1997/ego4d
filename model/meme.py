import torch
import torch.nn as nn
import torch.nn.functional as F

class MEME(nn.Module):
    def __init__(self, args):
        super(MEME, self).__init__()
        self.hidden_size = args.hidden_size
        self.embedding_dim = args.embedding_dim

        self.model = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 3) # 3 classes - confidence, start, end
        )
        
    def forward(self, x):
        output = self.model(x)
        output = F.softmax(output, dim=0)
        return output
