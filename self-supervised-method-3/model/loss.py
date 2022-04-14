import torch
import scipy.signal
import scipy.signal.windows
import torch.nn as nn
import torch.nn.functional as F

class MEME_SS_LOSS(nn.Module):
    def __init__(self, args):
        super(MEME_SS_LOSS, self).__init__()
        self.binCrossEnt = nn.BCEWithLogitsLoss()
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.L1Loss(reduction=args.reduction)
        self.cosineSim = nn.CosineEmbeddingLoss(reduction=args.reduction)
        
    def forward(self, pred, target_in_range, loss_type):
        output = self.pos_loss(pred, target_in_range, loss_type)
        return output

    def pos_loss(self, pred, groundTruth, loss_type="BCE"):
        '''
        Compute the POS style loss with tag prediction
        '''
        loss = None
        if loss_type == "BCE":
            loss = self.binCrossEnt(pred, groundTruth)
        elif loss_type == "Distance":
            loss = self.L1Loss(pred, groundTruth)
        elif loss_type == "Cosine":
            target = torch.ones(groundTruth.shape[0]).to(pred.device)
            loss = self.cosineSim(pred, groundTruth, target )
        return loss
