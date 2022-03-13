import torch
import scipy.signal
import scipy.signal.windows
import torch.nn as nn
import torch.nn.functional as F

class MEME_LOSS(nn.Module):
    def __init__(self, args):
        super(MEME_LOSS, self).__init__()
        self.loss_fn = nn.BCELoss()
        self.boundary_smoothing = args.boundary_smoothing
        
    def forward(self, pred, target_start, target_end, target_in_range,loss_type='pos_loss'):
        if loss_type == 'hard_qa':
            output = self.em_joint_loss(pred, target_start, target_end, target_in_range)
        elif loss_type == 'soft_qa':
            output = self.qa_soft_loss(pred, target_start, target_end, target_in_range)
        elif loss_type == 'pos_loss':
            output = self.pos_loss(pred, target_in_range)
        return output

    def pos_loss(self, pred, target_in_range):
        '''
        Compute the POS style loss with tag prediction
        '''
        pred_scores = pred[:,:,2]
        loss = self.loss_fn(pred_scores.reshape(-1), target_in_range.reshape(-1))
        return loss

    def em_joint_loss(self, pred, target_start, target_end, target_in_range):
        """
        Compute the EM loss for joint prediction.
        pred: (batch_size, seq_len, 3)
        target_start: (batch_size,seq_len)
        target_end: (batch_size,seq_len)
        """
        loss = 0.5 * self.loss_fn(pred[:,:,0].reshape(-1), target_start.reshape(-1)) + \
                0.5 * self.loss_fn(pred[:,:, 1].reshape(-1), target_end.reshape(-1))

        return loss

    def qa_soft_loss(self, pred, target_start, target_end, target_in_range):
        '''
        Compute the QA style loss with soft boundaries

        assumes that the target_start and target_end are (N) shaped
        '''
        #soften the bondaries using gaussian filter
        target_start = target_start.unsqueeze(0).to(float)
        target_end = target_end.unsqueeze(0).to(float)
        filter = torch.tensor(scipy.signal.windows.gaussian(self.boundary_smoothing, std=1)).unsqueeze(0).unsqueeze(0).to(float).to(pred.device)
        padding = int((self.boundary_smoothing -1) / 2)
        target_start = F.conv1d(target_start, filter, padding=padding).squeeze(0)
        target_end = F.conv1d(target_end, filter, padding=padding).squeeze(0)
        # target_start = F.softmax(target_start, dim=1)
        # target_end = F.log_softmax(target_end, dim=1)

        # loss = -torch.sum(F.log_softmax(pred[:, 0]) * target_start, dim=1)*0.5
        # loss += -torch.sum(F.log_softmax(pred[:, 1]) * target_end, dim=1)*0.5

        loss = -torch.sum(torch.log(pred[:,:,0]).reshape(-1) * target_start.reshape(-1))*0.5
        loss += -torch.sum(torch.log(pred[:,:,1]).reshape(-1) * target_end.reshape(-1))*0.5
        return torch.mean(loss)
