import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import init_custom_model, VIDEO_TOKEN, AUDIO_TOKEN, QUERY_TOKEN, EOS_TOKEN
from model.data_loader import Modal

class MEME_SS_BASE(nn.Module):
    def __init__(self, args):
        super(MEME_SS_BASE, self).__init__()
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

        self.reorder_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size ,bias=True),
        )

        self.frame_prediction = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(in_features=self.hidden_size, out_features=1, bias=True),
            nn.Softmax(dim=0),
        )

        self.frame_number_prediction = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(in_features=self.hidden_size, out_features=1, bias=True)
        )

    def forward(self, video, text, audio, modalities=None, **kwargs):
        bs,l,_ = video.shape
        video = self.project_video(video)
        if Modal._Audio in modalities:
            audio = self.project_audio(audio)
        text = self.project_text(text)
        input_ = self.create_model_input(video, audio, text, modalities)
        output = self.model(inputs_embeds = input_)

        frame_pred = self.frame_prediction(output[0])
        #reorder_pred = self.reorder_head(output[0])
        frame_number_pred = self.frame_number_prediction(output[0])
        
        #print("Output: ", output)
        #output only for each video frame
        output = output[0][:,1:l+1,:] #1 for <bos>
        frame_pred = frame_pred[0][1:l+1,0]#1 for <bos>
        #reorder_pred = reorder_pred[0][1:l+1,0]#1 for <bos>
        frame_number_pred = frame_number_pred[0][1:l+1,0]
        
        reorder_pred = None

        return output, frame_pred, reorder_pred, frame_number_pred

    def set_special_tokens(self):
        """VIDEO_TOKEN, AUDIO_TOKEN"""
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        vid = self.tokenizer.vocab[VIDEO_TOKEN]
        aud = self.tokenizer.vocab[AUDIO_TOKEN]
        eos = self.tokenizer.vocab[EOS_TOKEN]
        query = self.tokenizer.vocab[QUERY_TOKEN]
        self.special_tokens = torch.tensor([bos,sep,vid,aud,query,eos]).to(self.device)
        

    def get_token_types(self, video, audio, text):
        v, a, t = video.shape[1], audio.shape[1], text.shape[1]
        types = [0]*(v+1)+[1]*(a+1)+[2]*(t+1)+[2] #<bos>,<video>video,<audio>audio,<query>query,<query>query<eos>
        types = torch.tensor(types).to(self.device)
        return types

    def create_model_input(self, video, audio, text, modalities=None):
        a=3
        b=4
        embedding_layer = self.model.embeddings
        types = self.get_token_types(video, audio, text)
        special_token_embeds = embedding_layer.word_embeddings(self.special_tokens).unsqueeze(0) #[bos,sep,query,eos]
        token_type_embeds = embedding_layer.token_type_embeddings(types).unsqueeze(0)

        bos = special_token_embeds[:,0,:].unsqueeze(1)
        sep = special_token_embeds[:,1,:].unsqueeze(1)
        vid = special_token_embeds[:,2,:].unsqueeze(1)
        aud = special_token_embeds[:,3,:].unsqueeze(1)
        query = special_token_embeds[:,2,:].unsqueeze(1)
        eos = special_token_embeds[:,3,:].unsqueeze(1)
        input_embed = torch.cat([bos,video,sep,audio,query,text,eos],dim=1)
        model_input = input_embed + token_type_embeds

        return model_input

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
