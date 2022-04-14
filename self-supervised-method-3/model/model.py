import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import init_custom_model, VIDEO_TOKEN, AUDIO_TOKEN, QUERY_TOKEN, EOS_TOKEN
from model.data_loader import Modal
from collections import defaultdict

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
        self.mask = None

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
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.Tanh(),
            nn.Linear(in_features=self.hidden_size*2, out_features=1, bias=True)
        )

    def forward(self, video, text, text_length, audio, args, modalities=None, **kwargs):
        bs,l,_ = video.shape
        video = self.project_video(video)
        if Modal._Audio in modalities:
            audio = self.project_audio(audio)
        
        input_, clsTokens, masks = self.create_model_input(video, audio, text, text_length, args, modalities)

        output = self.model(inputs_embeds = input_)

        # frame_pred = self.frame_prediction(output[0])
        # #reorder_pred = self.reorder_head(output[0])
        # frame_number_pred = self.frame_number_prediction(output[0])

        # #print("Output: ", output)
        # #output only for each video frame
        # output = output[0][:,1:l+1,:] #1 for <bos>
        # frame_pred = frame_pred[0][1:l+1,0]#1 for <bos>
        # reorder_pred = reorder_pred[0][1:l+1,0]#1 for <bos>


        return output, clsTokens, masks
        #return output, frame_pred, reorder_pred, frame_number_pred_1, clsTokens, masks


    def set_special_tokens(self):
        """VIDEO_TOKEN, AUDIO_TOKEN"""
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        vid = self.tokenizer.vocab[VIDEO_TOKEN]
        aud = self.tokenizer.vocab[AUDIO_TOKEN]
        eos = self.tokenizer.vocab[EOS_TOKEN]
        query = self.tokenizer.vocab[QUERY_TOKEN]
        mask = self.tokenizer.mask_token_id
        self.special_tokens = torch.tensor([bos,sep,vid,aud,query,eos,mask]).to(self.device)
        

    def get_token_types(self, video, audio, text_length):
        v, a, t = video.shape[1], audio.shape[1], text_length
        types = [0]*(v+1)+[1]*(a+1)+[2]*(t+1)+[2] #<bos>,<video>video,<audio>audio,<query>query,<query>query<eos>
        types = torch.tensor(types).to(self.device)
        position = torch.tensor(range(len(types))).to(self.device)
        return types, position

    def create_model_input(self, video, audio, texts, text_length, args, modalities=None):
        embedding_layer = self.model.embeddings
        types, position = self.get_token_types(video, audio, text_length+len(texts))
        special_token_embeds = embedding_layer.word_embeddings(self.special_tokens).unsqueeze(0) #[bos,sep,vid,aud,query,eos]
        token_type_embeds = embedding_layer.token_type_embeddings(types).unsqueeze(0)
        position_embeds = embedding_layer.position_embeddings(position).unsqueeze(0)

        bos = special_token_embeds[:,0,:].unsqueeze(1)
        sep = special_token_embeds[:,1,:].unsqueeze(1)
        vid = special_token_embeds[:,2,:].unsqueeze(1)
        aud = special_token_embeds[:,3,:].unsqueeze(1)
        query = special_token_embeds[:,4,:].unsqueeze(1)
        eos = special_token_embeds[:,5,:].unsqueeze(1)
        mask = special_token_embeds[:,6,:].unsqueeze(1)
        _input = [bos, vid, video, aud, audio]
        startIndexs = defaultdict(list)
        startIndexs['Video'].append( bos.shape[1] + vid.shape[1] )
        startIndexs['Audio'].append( startIndexs['Video'][0] + video.shape[1] + aud.shape[1])  
        audio_mask_length = int(audio.shape[1]*.15)
        _audio_mask_index = torch.randperm(audio.shape[1])[:audio_mask_length]
        _audio_mask_index = torch.add(_audio_mask_index, startIndexs['Audio'][0])

        _text_mask_index = []
        clsTokens = defaultdict(list)
        temp_count = startIndexs['Audio'][0]+ audio.shape[1]
        for text in texts:
            _input.append(query)
            _text_proj = self.project_text(text.to(args.device))
            _input.append(_text_proj)
            temp_count += 1 
            startIndexs['Query'].append(temp_count)
            text_length_mask = int(_text_proj.shape[1]*.2)
            temp_text_mask_index = torch.randperm(_text_proj.shape[1])[:text_length_mask]
            _text_mask_index.append( torch.add(temp_text_mask_index, temp_count))
            temp_count += _text_proj.shape[1]

        _input.append(eos)
        
        cnt = -1
        for i, ten in enumerate(_input):
            cnt += ten.shape[1]
            if torch.equal(ten, vid):
                clsTokens['Video'].append(cnt)
                continue
            if torch.equal(ten, aud):
                clsTokens['Audio'].append(cnt)
                continue
            if torch.equal(ten, query):
                clsTokens['Query'].append(cnt)
                continue
        input_embed = torch.cat(_input,dim=1)

        _text_mask_index = torch.cat(_text_mask_index)

        masks = defaultdict()
        masks['Audio'] = _audio_mask_index
        masks['Query'] = _text_mask_index
        masks['AudioEmbedd'] = input_embed[0][_audio_mask_index]
        masks['QueryEmbedd'] = input_embed[0][_text_mask_index]
        
        if args.mask_audio:
            input_embed[0][masks['Audio']] = mask
        if args.mask_text:
            input_embed[0][masks['Query']] = mask

        model_input = input_embed + token_type_embeds + position_embeds

        return model_input, clsTokens, masks

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
