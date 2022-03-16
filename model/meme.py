import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_processing import Modal

class MEME(nn.Module):
    def __init__(self, args):
        super(MEME, self).__init__()
        self.hidden_size = args.hidden_size
        self.embedding_dim = args.embedding_dim

        self.model = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            # nn.Linear(int(self.hidden_size/2), 3) # 3 classes - confidence, start, end
        )
        self.start_head = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size/2), 1)
        )
        self.end_head = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size/2), 1)
        )
        self.ans_head = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size/2), 1)
        )
        self.apply(init_weights)
        
    def forward(self, features, query_emb, audio_features, modalities = None, infer=True):

        x = torch.cat((features, query_emb), dim=-1)
        if modalities is None or Modal._Audio in modalities:
            x = torch.cat((x, audio_features), dim=-1)

        output = self.model(x)
        start_output = self.start_head(output)
        end_output = self.end_head(output)
        ans_output = self.ans_head(output)*10
        
        if infer:
            start_output = F.softmax(start_output, dim=1)
            end_output = F.softmax(end_output, dim=1)
            ans_output = F.sigmoid(ans_output)

        output = torch.cat((start_output, end_output, ans_output), dim=-1)
        return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)