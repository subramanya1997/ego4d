import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import init_custom_model, QUERY_TOKEN, EOS_TOKEN
from utils.data_processing import Modal

class MEME_BASE(nn.Module):
    def __init__(self, args):
        super(MEME_BASE, self).__init__()
        model, tokenizer = init_custom_model()#TODO improve this
        self.hidden_size = model.config.hidden_size
        dropout = model.config.hidden_dropout_prob
        self.max_len = model.config.max_position_embeddings
        self.device = args.device

        self.model = model
        self.tokenizer = tokenizer
        self.set_special_tokens()

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
        
    def forward(self, video, text, audio, modalities=None, lengths = None, **kwargs):
        ## TRIM INPUT
        if video.shape[1]*2+5>self.max_len:
            win_size = int((self.max_len - 5)/2) - 1
            video = video[:,:win_size,:]
            audio = audio[:,:win_size,:]

        bs,l,_ = video.shape
        video = self.project_video(video)
        if Modal._Audio in modalities:
            audio = self.project_audio(audio)
        text = text[:,0,:].unsqueeze(dim=1)
        text = self.project_text(text)
        input_ = self.create_model_input(video, audio, text, lengths)
        output = self.model(inputs_embeds = input_)[0]

        #output only for each video frame
        output = output[:,1:l+1,:] #1 for <bos>
        return output

    def set_special_tokens(self):
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        eos = self.tokenizer.vocab[EOS_TOKEN]
        query = self.tokenizer.vocab[QUERY_TOKEN]
        pad = self.tokenizer.pad_token_id
        self.special_tokens = torch.tensor([bos,sep,query,eos,pad]).to(self.device)

    def get_token_types_and_position(self, video, audio, text, lengths):
        v, a, t = video.shape[1], audio.shape[1], text.shape[1]
        types_ = [0]*(v+1)+[1]*(a+1)+[2]*(t+1)+[2] #<bos>,<sep>video,<sep>audio,<query>query,<eos>
        position = torch.tensor(range(len(types_))).to(self.device).repeat(video.shape[0],1)
        types = torch.tensor(types_).to(self.device).repeat(video.shape[0],1)

        # l_last = lengths[-1]
        # types_last = [2]*types.shape[1]
        # types_last_ = [0]*(l_last+1)+[1]*(l_last+1)+[2]*(t+1)+[2] #<bos>,<sep>video,<sep>audio,<query>query,<eos>
        # types_last[:len(types_last_)] = types_last_
        # types_last = torch.tensor(types_last).to(self.device).unsqueeze(0)
        # print(types_last.shape, types.shape, lengths)
        # types[-1] = types_last
        #fix types of the last one

        return types, position

    def create_model_input(self, video, audio, text, lengths, modalities=None):
        bs = video.shape[0]

        embedding_layer = self.model.roberta.embeddings
        types, position = self.get_token_types_and_position(video, audio, text, lengths) 
        special_token_embeds = embedding_layer.word_embeddings(self.special_tokens).unsqueeze(0).repeat(bs,1,1) #[bos,sep,query,eos,pad]
        token_type_embeds = embedding_layer.token_type_embeddings(types)
        position_embeds = embedding_layer.position_embeddings(position)

        bos = special_token_embeds[:,0,:].unsqueeze(1)
        sep = special_token_embeds[:,1,:].unsqueeze(1)
        query = special_token_embeds[:,2,:].unsqueeze(1)
        eos = special_token_embeds[:,3,:].unsqueeze(1)
        pad = special_token_embeds[:1,4,:].unsqueeze(1)

        # make same batch size
        
        input_embed_a = torch.cat([bos[:-1],video[:-1],sep[:-1],audio[:-1],query[:-1],text[:-1],eos[:-1]],dim=1)
        input_embed_b = torch.cat([bos[-1:],video[-1:,:lengths[-1]],sep[-1:],audio[-1:,:lengths[-1]],query[-1:],text[-1:],eos[-1:]],dim=1)
        # pad input_embed_b to input_embed_a shape
        pad_length = input_embed_a.shape[1]-input_embed_b.shape[1]
        padding = pad.repeat(1,pad_length,1)
        input_embed_b = torch.cat([input_embed_b,padding],dim=1)
        input_embed = torch.cat([input_embed_a,input_embed_b],dim=0)
        model_input = input_embed + token_type_embeds + position_embeds

        return model_input
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)