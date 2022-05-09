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
        # self.project_audio = nn.Sequential(
        #     nn.Linear(args.audio_feature_size, self.hidden_size),
        #     nn.LayerNorm(self.hidden_size),
        #     nn.Dropout(dropout),
        # )
        self.project_text = nn.Sequential(
            nn.Linear(args.query_feature_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
        )
        
    def forward(self, input_list, modalities = None, window_labels = False, **kwargs):
        input_ = []
        max_input_len = 0
        for data_ in input_list:
            (_, _, _, _, _, _, info, _) = data_
            if max_input_len < (info['video_text_length'] + 3):
                max_input_len = info['video_text_length'] + 3
        for data_ in input_list:
            (sample_id, clip_id, video, audio, query_emb, answer, info, lens) = data_
            
            video = self.project_video(video)
            query_emb = self.project_text(query_emb)

            input_.append( self.create_model_input(video, query_emb, lens, max_input_len) )

        input_ = torch.cat(input_,dim=0)

        output = self.model(inputs_embeds = input_)
        start_logits = output.start_logits
        end_logits = output.end_logits

        return start_logits, end_logits

    def set_special_tokens(self):
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        eos = self.tokenizer.vocab[EOS_TOKEN]
        query = self.tokenizer.vocab[QUERY_TOKEN]
        pad = self.tokenizer.pad_token_id
        self.special_tokens = torch.tensor([bos,sep,query,eos,pad]).to(self.device)

    def get_token_types_and_position(self, input_types_length):
        types_ = [0] * input_types_length[0] + [1] * input_types_length[1] + [2] * input_types_length[2] + [3] * input_types_length[3] + [4] * input_types_length[4] +  [5] * input_types_length[5]
        position_ = [0] + [1] * input_types_length[1]
        for i in range(input_types_length[3]+2):
            position_.append(i+2)
        _temp_p = position_[-1]
        for i in range(input_types_length[5]):
            position_.append(_temp_p+1)
        
        types = torch.tensor(types_).to(self.device)
        position = torch.tensor(position_).to(self.device)
        return types, position

    def create_model_input(self, video, text, lengths, max_input_len):
        bs = video.shape[0]
        embedding_layer = self.model.roberta.embeddings
        special_token_embeds = embedding_layer.word_embeddings(self.special_tokens).unsqueeze(0).repeat(bs,1,1)
        
        bos = special_token_embeds[:,0,:].unsqueeze(1)
        sep = special_token_embeds[:,1,:].unsqueeze(1)
        query = special_token_embeds[:,2,:].unsqueeze(1)
        eos = special_token_embeds[:,3,:].unsqueeze(1)
        pad = special_token_embeds[:1,4,:].unsqueeze(1)

        input_embed = [bos,video,query,text,eos]
        input_embed_length = bos.shape[1] + video.shape[1] + query.shape[1] + text.shape[1] + eos.shape[1]
        for i in range(max_input_len-input_embed_length):
            input_embed.append(pad)

        input_types_length = [bos.shape[1], video.shape[1], query.shape[1], text.shape[1], eos.shape[1], max_input_len-input_embed_length]
        input_types, input_position = self.get_token_types_and_position(input_types_length)

        token_type_embeds = embedding_layer.token_type_embeddings(input_types)
        position_embeds = embedding_layer.position_embeddings(input_position)
        
        input_embed = torch.cat(input_embed,dim=1)

        model_input = input_embed + token_type_embeds + position_embeds

        return model_input
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)