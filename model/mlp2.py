import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_processing import Modal

class MLP2(nn.Module):
    def __init__(self, args):
        super(MLP2, self).__init__()
        self.hidden_size = args.hidden_size
        self.window_size = args.clip_window
        dropout = args.dropout

        self.input_size = self.hidden_size*self.window_size * 2 + args.query_feature_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, int(self.hidden_size*self.window_size/4)),
            nn.LayerNorm(int(self.hidden_size*self.window_size/4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.hidden_size*self.window_size/4), self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.window_size) # 3 classes - confidence, start, end
        )
        self.project_video = nn.Sequential(
            nn.Linear(args.video_feature_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
        )
        self.project_audio = nn.Sequential(
            nn.Linear(args.audio_feature_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
        )
        self.project_text = nn.Sequential(
            nn.Linear(args.query_feature_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
        )

        self.type_embed = nn.Embedding(3, self.hidden_size)

        self.apply(init_weights)
        
    def get_type_embeds(self, video, text, audio):
        bs,_,_ = video.shape
        tokens = [0]*video.shape[1] + [1]*audio.shape[1] + [2]*text.shape[1]
        tokens = torch.tensor(tokens).repeat(1,1).to(video).to(torch.long)
        tokens = self.type_embed(tokens).reshape(bs,-1)
        return tokens

    def pad_tokens(self, tensor):
        bs,len = tensor.shape
        pad_len = self.input_size - len
        padding = torch.zeros(bs,pad_len).to(tensor)
        tensor = torch.cat([tensor,padding],dim=-1)
        return tensor

    def forward(self, video, text, audio, modalities = None, infer=True):
        video = self.project_video(video)[:,:self.window_size,:]
        audio = self.project_audio(audio)[:,:self.window_size,:]

        text = text[:,0,:].unsqueeze(1)
        text = self.project_text(text)

        token_types = self.get_type_embeds(video, text, audio)

        bs,l,_ = video.shape
        video = video.reshape(bs,-1)
        audio = audio.reshape(bs,-1)
        text = text[:,0,:].reshape(bs,-1)
        if modalities is None or Modal._Audio in modalities:
            x = torch.cat((video, audio, text), dim=-1)
        else:
            x = torch.cat((video, text), dim=-1)

        x = self.pad_tokens(x)
        token_types = self.pad_tokens(token_types)

        x = x+token_types
        output = self.model(x)
        
        if infer:
            output = F.sigmoid(output)
        output = output[:,:l].unsqueeze(-1).repeat(1,1,3)
        return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)