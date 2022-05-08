import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import init_custom_model, QUERY_TOKEN, EOS_TOKEN
from utils.data_processing import Modal
from model.meme_loss import MEME_LOSS

class MEME_B(nn.Module):
    def __init__(self, args):
        super(MEME_B, self).__init__()
        model, tokenizer = init_custom_model()#TODO improve this
        self.hidden_size = model.config.hidden_size
        dropout = model.config.hidden_dropout_prob
        self.max_len = model.config.max_position_embeddings
        self.device = args.device
        self.loss_fn = MEME_LOSS(args)
        self.loss_wt = args.loss_weight3 if hasattr(args, 'loss_weight3') else 0.5

        self.model = model
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(self.hidden_size, 2)
        self.qa_classifier = nn.Linear(self.hidden_size, 2)
        self.center_classifier = nn.Linear(self.hidden_size, 1)

        self.set_special_tokens()

        self.project_video = nn.Sequential(
            nn.Linear(args.video_feature_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
        )
        
    def forward(self, video, text, audio, clip_lengths = None, query_lengths = None, \
                    center_idx = None, is_ans = None, starts = None, ends = None, **kwargs):
        bs,l,_ = video.shape
        video = self.project_video(video)
        input_ = self.create_model_input(video, audio, text, clip_lengths, query_lengths)
        roberta_output = self.model(inputs_embeds = input_)[0]
        output = self.center_classifier(roberta_output)

        #output only for each video frame
        losses = []
        center_idx = center_idx.to(torch.long)
        prediction = []
        for i in range(bs):
            t = query_lengths[i]
            l = clip_lengths[i]
            predictions = output[i,t+2:t+l+2,0]
            loss = self.loss_fn(predictions.unsqueeze(0), starts, ends, is_ans, loss_type = 'center', center_idx = center_idx[i])
            losses.append(loss)
            prediction.append(predictions)
        loss = torch.stack(losses)

        return output, loss

    def loss(self, pred, starts, ends, is_ans, loss_type = 'joint_loss'):
        return self.loss_fn(pred, starts, ends, is_ans, loss_type = loss_type)

    def set_special_tokens(self):
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        eos = self.tokenizer.vocab[EOS_TOKEN]
        query = self.tokenizer.vocab[QUERY_TOKEN]
        pad = self.tokenizer.pad_token_id
        self.special_tokens = torch.tensor([bos,sep,query,eos,pad]).to(self.device)

    def get_token_types_and_position(self, video, audio, text, clip_lengths, query_lengths):
        v, a, t = video.shape[1], audio.shape[1], text.shape[1]
        total_length = v + t + 3
        types_ = [[0]*(q+1)+[1]*(c+1)+[2]*(total_length - (q+c+2)) for c,q in zip(clip_lengths,query_lengths)] #<bos>,<sep>video,<sep>audio,<query>query,<eos>
        types = torch.tensor(types_, dtype=torch.long, device=video.device)
        
        position = torch.tensor(range(total_length)).to(video.device).repeat(video.shape[0],1)

        return types, position

    def concat_stuff(self, video, audio, text, clip_lengths, query_lengths, special_token_embeds):
        bos = special_token_embeds[:,0,:]
        sep = special_token_embeds[:,1,:]
        query = special_token_embeds[:,2,:]
        eos = special_token_embeds[:,3,:]
        pad = special_token_embeds[:,4,:]

        # concat according to lengths
        b,v,_ = video.shape
        t = text.shape[1]
        input_embed = torch.zeros(b,v+t+3, self.hidden_size, device = video.device)
        input_embed[:,0,:] = bos
        for i in range(b):
            input_embed[i,1:query_lengths[i]+1,:] = text[i,:query_lengths[i],:]
            input_embed[i,query_lengths[i]+1,:] = sep[i]
            input_embed[i,query_lengths[i]+2:query_lengths[i]+2+clip_lengths[i],:] = video[i,:clip_lengths[i],:]
            input_embed[i,query_lengths[i]+2+clip_lengths[i],:] = eos[i]
            input_embed[i,query_lengths[i]+2+clip_lengths[i]+1:,:] = pad[i]

        return input_embed


    def create_model_input(self, video, audio, text, clip_lengths, query_lengths):
        bs = video.shape[0]

        embedding_layer = self.model.embeddings
        types, position = self.get_token_types_and_position(video, audio, text, clip_lengths, query_lengths) 
        special_token_embeds = embedding_layer.word_embeddings(self.special_tokens.to(video.device)).unsqueeze(0).repeat(bs,1,1) #[bos,sep,query,eos,pad]
        token_type_embeds = embedding_layer.token_type_embeddings(types)
        position_embeds = embedding_layer.position_embeddings(position)

        input_embed = self.concat_stuff(video, audio, text, clip_lengths, query_lengths, special_token_embeds)
        model_input = input_embed + token_type_embeds + position_embeds

        return model_input
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)