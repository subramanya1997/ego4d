import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import init_custom_model, QUERY_TOKEN, EOS_TOKEN
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

        print(model)

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
            nn.Linear(in_features=self.hidden_size,
                      out_features=2000,bias=True),
            nn.BatchNorm1d(2000), 
            nn.ReLU(),
            nn.Linear(in_features=2000,
                      out_features=1000),
        )

        self.frame_prediction = nn.Sequential(
            nn.Linear(in_features=self.hidden_size,
                      out_features=300,bias=True),
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.hidden_size, 5)

    def forward(self, video, text, audio, modalities=None, **kwargs):
        bs,l,_ = video.shape
        video = self.project_video(video)
        if Modal._Audio in modalities:
            audio = self.project_audio(audio)
        text = self.project_text(text)
        input_ = self.create_model_input(video, audio, text, modalities)
        output = self.model(inputs_embeds = input_)

        print(input_.shape, output[0].shape, output[0][:, 0, :].shape)
        #frame prediction head

        x = output[0][:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        frame_pred = self.frame_prediction(output[0])
        
        

        #print("Output: ", output)
        #output only for each video frame
        output = output[0][:,1:l+1,:] #1 for <bos>

        return output, x

    def set_special_tokens(self):
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        eos = self.tokenizer.vocab[EOS_TOKEN]
        query = self.tokenizer.vocab[QUERY_TOKEN]
        self.special_tokens = torch.tensor([bos,sep,query,eos]).to(self.device)
        

    def get_token_types(self, video, audio, text):
        v, a, t = video.shape[1], audio.shape[1], text.shape[1]
        types = [0]*(v+1)+[1]*(a+1)+[2]*(t+1)+[2] #<bos>,<sep>video,<sep>audio,<query>query,<eos>
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
        query = special_token_embeds[:,2,:].unsqueeze(1)
        eos = special_token_embeds[:,3,:].unsqueeze(1)
        input_embed = torch.cat([bos,video,sep,audio,query,text,eos],dim=1)
        model_input = input_embed + token_type_embeds

        return model_input

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)