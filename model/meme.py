import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mlp import MLP
from model.mlp2 import MLP2
from model.meme_transformer import MEME_BASE

from utils.data_processing import Modal

class MEME(nn.Module):
    def __init__(self, args):
        super(MEME, self).__init__()
        if args.model=='MLP':
            self.model=MLP(args)
        elif args.model=='MLP2':
            self.model=MLP2(args)
        else:
            self.model=MEME_BASE(args)
        
    def forward(self, input_list, **kwargs):
        output = self.model(input_list, **kwargs)
        return output
