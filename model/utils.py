import torch

from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizerFast

def init_custom_model(folder_path="output/models/model1", model_name= "meme"):
    my_config = RobertaConfig(num_hidden_layers=2, max_position_embeddings=2048,type_vocab_size=3)
    my_config.save_pretrained(save_directory=folder_path)
    my_config = RobertaConfig.from_pretrained(f"{folder_path}/config.json")

    model = RobertaForTokenClassification(my_config)

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    tokenizer.train(files=[f"{folder_path}/dummy_text.txt"], vocab_size=256, min_frequency=1,
                    show_progress=True,
                    special_tokens=["<s>", "<sep>", "<pad>", "</s>", "<unk>", "<mask>", "<eos>"])
    #Save the Tokenizer to disk
    tokenizer.save_model(folder_path)

    fast_tokenizer = RobertaTokenizerFast.from_pretrained(folder_path)
    model.resize_token_embeddings(len(fast_tokenizer))
    torch.save(model.state_dict(), f'{folder_path}/{model_name}.pt')
    model.save_pretrained(save_directory=folder_path)

    return model, fast_tokenizer
