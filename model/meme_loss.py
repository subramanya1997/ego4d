import torch
import torch.nn as nn

class MEME_LOSS(nn.Module):
    def __init__(self, args):
        super(MEME_LOSS, self).__init__()
        self.loss_fn = nn.BCELoss()
        
    def forward(self, pred, target_start, target_end, target_in_range):
        output = self.em_joint_loss(pred, target_start, target_end, target_in_range)
        return output

    def em_joint_loss(self, pred, target_start, target_end, target_in_range):
        """
        Compute the EM loss for joint prediction.
        pred: (batch_size, 3)
        target_start: (batch_size,)
        target_end: (batch_size,)
        """
        loss = 0.5 * self.loss_fn(pred[:, 0], target_start) + \
                0.5 * self.loss_fn(pred[:, 1], target_end)

        return torch.mean(loss)