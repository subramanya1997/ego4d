import torch
import random
import numpy as np

import torch.nn.functional as F

from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizerFast

QUERY_TOKEN = "<query>"
EOS_TOKEN = "<eos>"

def init_custom_model(folder_path="output/models/model1", model_name= "meme"):
    my_config = RobertaConfig(num_hidden_layers=2, max_position_embeddings=2048,type_vocab_size=6)
    my_config.save_pretrained(save_directory=folder_path)
    my_config = RobertaConfig.from_pretrained(f"{folder_path}/config.json")

    model = RobertaForTokenClassification(my_config)

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    tokenizer.train(files=[f"{folder_path}/dummy_text.txt"], vocab_size=256, min_frequency=1,
                    show_progress=True,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    #Save the Tokenizer to disk

    tokenizer.save_model(folder_path)

    fast_tokenizer = RobertaTokenizerFast.from_pretrained(folder_path)
    num_added_toks = fast_tokenizer.add_tokens([QUERY_TOKEN], special_tokens=True)
    num_added_toks = fast_tokenizer.add_tokens([EOS_TOKEN], special_tokens=True)
    fast_tokenizer.save_pretrained(folder_path)
    model.resize_token_embeddings(len(fast_tokenizer))
    torch.save(model.state_dict(), f'{folder_path}/{model_name}.pt')
    model.save_pretrained(save_directory=folder_path)

    return model, fast_tokenizer

def fix_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

def make_windows(x, window_size,pad_value=0.):
    x = torch.split(x, window_size, dim=1)
    x = list(x)
    lens = [y.shape[1] for y in x]
    if len(x[0].shape) == 2:
        x[-1] = F.pad(x[-1],pad=(0,window_size-x[-1].shape[1]),value=pad_value)
    else:
        x[-1] = F.pad(x[-1],pad=(0,0,0,window_size-x[-1].shape[1]),value=pad_value)
    x = torch.cat(x,dim=0)
    return x, lens

def label_windows(is_ans):
    # is_ans.shape = (batch_size, window_size)
    _, w_s = is_ans.shape
    has_ans = is_ans.sum(dim=1)
    labels = has_ans > 0
    return labels

