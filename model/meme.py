import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mlp import MLP
from model.mlp2 import MLP2
from model.meme_transformer import MEME_BASE
from model.meme_multi import MEME_MULTI
from model.meme_b import MEME_B

from utils.data_processing import Modal

class MEME(nn.Module):
    def __init__(self, args):
        super(MEME, self).__init__()
        if args.model=='MLP':
            self.model=MLP(args)
        elif args.model=='MLP2':
            self.model=MLP2(args)
        elif args.model=='MEME_MULTI':
            self.model=MEME_MULTI(args)
        elif args.model=='MEME_B':
            self.model=MEME_B(args)
        else:
            self.model=MEME_BASE(args)
        
    def forward(self, features, query_emb, audio_features, **kwargs):
        output = self.model(features, query_emb, audio_features, **kwargs)
        return output
