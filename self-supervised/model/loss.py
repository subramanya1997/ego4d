import torch
import scipy.signal
import scipy.signal.windows
import torch.nn as nn
import torch.nn.functional as F

class MEME_SS_LOSS(nn.Module):
    def __init__(self, args):
        #self.loss_fn = nn.BCELoss()
        pass