"""MEME
"""
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel
import torch.nn.functional as F



def build_optimizer_and_scheduler(model, configs):
    no_decay = [
        "bias",
        "layer_norm",
        "LayerNorm",
    ]  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        configs.num_train_steps * configs.warmup_proportion,
        configs.num_train_steps,
    )
    return optimizer, scheduler

class HighLightLayer(nn.Module):
    def __init__(self, dim, configs):
        super(HighLightLayer, self).__init__()
        config = RobertaConfig(num_hidden_layers=12, max_position_embeddings=2048, add_cross_attention=False, is_decoder=False)
        self.encoder = RobertaModel(config=config, add_pooling_layer = False)
        hidden_size = self.encoder.config.hidden_size
        dropout = configs.drop_rate
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, mask):
        # compute logits
        logits = self.encoder(inputs_embeds=x, attention_mask=mask)[0]
        logits = self.head(logits)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction="none")(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = nn.Sigmoid()(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        inputs = nn.Sigmoid()(inputs)  
        targets = targets.type(torch.float32)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class MEME(nn.Module):
    def __init__(self, configs, word_vectors=None):
        super(MEME, self).__init__()
        self.configs = configs

        self.encoder_config = RobertaConfig(num_hidden_layers=12, max_position_embeddings=800, hidden_size=1024, num_attention_heads=16, add_cross_attention=False, is_decoder=False)
        self.decoder_config = RobertaConfig(num_hidden_layers=12, max_position_embeddings=800, hidden_size=1024, num_attention_heads=16, add_cross_attention=True, is_decoder=True)

        self.text_features = RobertaModel.from_pretrained("roberta-large", add_pooling_layer = False)
        # for param in self.text_features.parameters():
        #     param.requires_grad = False
        
        # self.feature_encoder = RobertaModel.from_pretrained("roberta-large", add_pooling_layer = False)

        self.feature_encoder = RobertaModel(config=self.encoder_config, add_pooling_layer = False)

        self.feature_decoder = RobertaModel(config=self.decoder_config, add_pooling_layer = False)

        self.hidden_size = self.feature_encoder.config.hidden_size

        dropout = configs.drop_rate

        self.project_video = nn.Sequential(
            nn.Linear(configs.video_feature_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
        )

        # query-guided highlighting
        # self.highlight_layer = HighLightLayer(dim=configs.dim, configs=configs)

        self.QA_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 2)
        )

        self.highlight_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1)
        )

        self.IoUloss = IoULoss()
        self.FocalLoss = FocalLoss()

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):

        video_features = self.project_video(video_features)
        query_features = self.text_features(input_ids=word_ids['input_ids'], attention_mask=word_ids['attention_mask'], token_type_ids=word_ids['token_type_ids'])[0]

        features = torch.cat((video_features, query_features), 1)
        masks = torch.cat((v_mask, q_mask), 1).to(torch.int32)
        token_types = torch.cat(( torch.zeros_like(v_mask), torch.ones_like(q_mask)), 1).to(torch.int32)

        features = self.feature_encoder(inputs_embeds=features, attention_mask=masks, token_type_ids=token_types)[0]
        # query_features = self.feature_encoder(inputs_embeds=query_features, attention_mask=q_mask)[0]

        h_score_1 = self.highlight_layer(features[:,:video_features.shape[1],:])
        temp_features = features[:,:video_features.shape[1],:] * h_score_1

        features = self.feature_decoder(inputs_embeds=temp_features, attention_mask=v_mask, encoder_hidden_states=query_features, encoder_attention_mask=q_mask)[0]

        # features = self.feature_decoder(inputs_embeds=features[:,:video_features.shape[1],:], attention_mask=v_mask, encoder_hidden_states=query_features, encoder_attention_mask=q_mask)[0]

        h_score = self.highlight_layer(features)

        features = features * h_score
        logits = self.QA_head(features)
        h_score = h_score.squeeze(-1)
        h_score_1 = h_score_1.squeeze(-1)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # return None, start_logits, end_logits

        return h_score_1, h_score, start_logits, end_logits

    def extract_index(self, start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)

        # _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
        # _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
        # return start_index, end_index

        # Get top 5 start and end indices.
        batch_size, height, width = outer.shape
        outer_flat = outer.view(batch_size, -1)
        temp, flat_indices = outer_flat.topk(5, dim=-1)
        start_indices = flat_indices // width
        end_indices = flat_indices % width
        return start_indices, end_indices

    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(
            scores=scores, labels=labels, mask=mask
        )

    def compute_cross_entropy_loss(self, start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction="mean")(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction="mean")(end_logits, end_labels)
        return start_loss + end_loss

    def compute_f_iou_loss(self, logits, label):
        iou_loss = self.IoUloss(logits, label)
        focal_loss = self.FocalLoss(logits, label)
        return iou_loss + focal_loss
