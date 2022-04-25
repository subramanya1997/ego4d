import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import init_custom_model, QUERY_TOKEN, EOS_TOKEN
from utils.data_processing import Modal
from model.meme_loss import MEME_LOSS

class MEME_MULTI(nn.Module):
    def __init__(self, args):
        super(MEME_MULTI, self).__init__()
        model, tokenizer = init_custom_model()#TODO improve this
        self.hidden_size = model.config.hidden_size
        dropout = model.config.hidden_dropout_prob
        self.max_len = model.config.max_position_embeddings
        self.device = args.device
        self.loss_fn = MEME_LOSS(args)

        self.model = model
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(self.hidden_size, 2)
        self.qa_classifier = nn.Linear(self.hidden_size, 2)
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
        
    def forward(self, video, text, audio, modalities=None, lengths = None, loss_labels=None, **kwargs):
        ## TRIM INPUT
        if video.shape[1]*2+5>self.max_len:
            win_size = int((self.max_len - 5)/2) - 1
            video = video[:,:win_size,:]
            audio = audio[:,:win_size,:]

        bs,l,_ = video.shape
        video = self.project_video(video)
        if Modal._Audio in modalities:
            audio = self.project_audio(audio)
        # text = text[:,0,:].unsqueeze(dim=1)
        text = self.project_text(text)
        input_ = self.create_model_input(video, audio, text, lengths)
        roberta_output = self.model(inputs_embeds = input_)[0]
        output = self.classifier(roberta_output)
        qa_output = self.qa_classifier(roberta_output)

        #output only for each video frame
        t = text.shape[1]
        offest = t+1
        output = output[:,offest:offest+l+1,:] #1st embedding for window label
        qa_output = qa_output[:,offest:offest+l+1,:]

        loss = self.loss_fn(output, loss_labels['starts'], loss_labels['ends'], loss_labels['is_ans'], loss_type = loss_labels['loss_type'])
        qa_loss = self.loss_fn(qa_output[:,1:], loss_labels['starts'], loss_labels['ends'], loss_labels['is_ans'], loss_type = 'hard_qa',lens = lengths)

        final_output = torch.cat([output, qa_output], dim=-1)
        return final_output, loss + qa_loss

    def loss(self, pred, starts, ends, is_ans, loss_type = 'joint_loss'):
        return self.loss_fn(pred, starts, ends, is_ans, loss_type = loss_type)

    def set_special_tokens(self):
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        eos = self.tokenizer.vocab[EOS_TOKEN]
        query = self.tokenizer.vocab[QUERY_TOKEN]
        pad = self.tokenizer.pad_token_id
        self.special_tokens = torch.tensor([bos,sep,query,eos,pad]).to(self.device)

    def get_token_types_and_position(self, video, audio, text, lengths):
        v, a, t = video.shape[1], audio.shape[1], text.shape[1]
        # types_ = [0]*(v+1)+[1]*(a+1)+[2]*(t+1)+[2] #<bos>,<sep>video,<sep>audio,<query>query,<eos>
        types_ = [0]*(t+1)+[1]*(v+1)+[2]*(a+1)+[2] #<bos>,<sep>video,<sep>audio,<query>query,<eos>
        position = torch.tensor(range(len(types_))).to(video.device).repeat(video.shape[0],1)
        types = torch.tensor(types_).to(video.device).repeat(video.shape[0],1)

        l_last = lengths[-1]
        types_last = [2]*types.shape[1]
        # types_last_ = [0]*(l_last+1)+[1]*(l_last+1)+[2]*(t+1)+[2] #<bos>,<sep>video,<sep>audio,<query>query,<eos>
        types_last_ = [0]*(t+1)+[1]*(l_last+1)+[2]*(l_last+1)+[2] #<bos>,<sep>video,<sep>audio,<query>query,<eos>
        types_last[:len(types_last_)] = types_last_
        types_last = torch.tensor(types_last).to(video.device).unsqueeze(0)
        types[-1] = types_last
        #fix types of the last one

        # pos_t = torch.tensor(range(t+1)).repeat(len(lengths),1)
        # pos = torch.tensor(range(t+1,t+1+(lengths[0]+1)*len(lengths)))
        # pos_v = pos.reshape(-1, lengths[0]+1)
        # pos_a = pos_v.clone()
        # eos_pos = torch.tensor(range(t+1+(lengths[0]+1),t+1+(lengths[0]+1)*len(lengths)+1,lengths[0]+1))
        # eos_pos = eos_pos.unsqueeze(-1)

        # print(t,v,a,lengths,pos_t.shape,pos_v.shape,eos_pos.shape)
        # pos = torch.cat([pos_t, pos_v, pos_a, eos_pos], dim=1)
        # pos[-1][t+1+lengths[-1]+1:t+1+(lengths[-1]+1)*2] = pos_a[-1][:(lengths[-1]+1)]
        # pos[-1][t+1+(lengths[-1]+1)*2:] = torch.tensor(range(pos_a[-1][(lengths[-1]+1)],\
        #     pos.shape[-1]-(t+1+(lengths[-1]+1)*2)+pos_a[-1][(lengths[-1]+1)]))
        # pos = torch.clamp_max(pos, self.max_len-1)
        # pos = pos.to(video.device)
        return types, position

    def create_model_input(self, video, audio, text, lengths, modalities=None):
        bs = video.shape[0]

        embedding_layer = self.model.embeddings
        types, position = self.get_token_types_and_position(video, audio, text, lengths) 
        special_token_embeds = embedding_layer.word_embeddings(self.special_tokens.to(video.device)).unsqueeze(0).repeat(bs,1,1) #[bos,sep,query,eos,pad]
        token_type_embeds = embedding_layer.token_type_embeddings(types)
        position_embeds = embedding_layer.position_embeddings(position)

        bos = special_token_embeds[:,0,:].unsqueeze(1)
        sep = special_token_embeds[:,1,:].unsqueeze(1)
        query = special_token_embeds[:,2,:].unsqueeze(1)
        eos = special_token_embeds[:,3,:].unsqueeze(1)
        pad = special_token_embeds[:1,4,:].unsqueeze(1)

        # make same batch size
        
        # input_embed_a = torch.cat([bos[:-1],video[:-1],sep[:-1],audio[:-1],query[:-1],text[:-1],eos[:-1]],dim=1)
        # input_embed_b = torch.cat([bos[-1:],video[-1:,:lengths[-1]],sep[-1:],audio[-1:,:lengths[-1]],query[-1:],text[-1:],eos[-1:]],dim=1)
        input_embed_a = torch.cat([bos[:-1],text[:-1],sep[:-1],video[:-1],sep[:-1],audio[:-1],eos[:-1]],dim=1)
        input_embed_b = torch.cat([bos[-1:],text[-1:],sep[-1:],video[-1:,:lengths[-1]],sep[-1:],audio[-1:,:lengths[-1]],eos[-1:]],dim=1)
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