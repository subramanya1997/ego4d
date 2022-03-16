import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import init_custom_model

class MEME_BASE(nn.Module):
    def __init__(self, args):
        super(MEME_BASE, self).__init__()
        model, tokenizer = init_custom_model()#TODO improve this
        self.hidden_size = model.config.hidden_size
        dropout = model.config.hidden_dropout_prob

        self.model = model
        self.tokenizer = tokenizer
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
        
    def forward(self, video, audio, text):
        video = self.project_video(video)
        audio = self.project_audio(audio)
        text = self.project_text(text)
        input = self.create_model_input(video, audio, text)


        return 0

    def create_model_input(self, video, audio, text):
        a=3
        b=4
        return 0
        

        

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)