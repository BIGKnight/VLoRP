from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model


def get_lora_model(base_model, rank = 16, lora_alpha = 16, target_modules = "all-linear"):
    lora_config = LoraConfig(r = rank, lora_alpha = lora_alpha, target_modules=target_modules)
    base_model.enable_input_require_grads()
    model = get_peft_model(base_model, lora_config)
    return model 


def model_constructor(model_name, model_max_length):
    if model_name.lower() == 'gpt2-xl':
        tokenizer = GPT2Tokenizer.from_pretrained('/path/to/model', model_max_length=model_max_length)
        model = GPT2LMHeadModel.from_pretrained('/path/to/model')
    elif model_name.lower() == 'gpt2-large':
        tokenizer = GPT2Tokenizer.from_pretrained(
            '/path/to/model', 
            model_max_length=model_max_length
        )
        model = GPT2LMHeadModel.from_pretrained('/path/to/model')
    elif model_name.lower() == 'gpt2-medium':
        tokenizer = GPT2Tokenizer.from_pretrained(
            '/path/to/model', 
            model_max_length=model_max_length
        )
        model = GPT2LMHeadModel.from_pretrained('/path/to/model')
    elif model_name.lower() == 'gpt2-small':
        tokenizer = GPT2Tokenizer.from_pretrained(
            '/path/to/model', 
            model_max_length=model_max_length
        )
        model = GPT2LMHeadModel.from_pretrained('/path/to/model')
    elif model_name.lower() == 'open_llama_3b':
        tokenizer = AutoTokenizer.from_pretrained(
            "/path/to/model", 
            model_max_length=model_max_length
        )
        model = LlamaForCausalLM.from_pretrained(
            "/path/to/model",
        )
    elif model_name.lower() == 'llama-3.1-8b':
        tokenizer = AutoTokenizer.from_pretrained(
            "/path/to/model", 
            model_max_length=model_max_length
        )
        model = LlamaForCausalLM.from_pretrained(
            "/path/to/model",
        )
    elif model_name.lower() == 'llama-2-7b':
        tokenizer = LlamaTokenizer.from_pretrained(
            "/path/to/model", 
            model_max_length=model_max_length
        )
        model = LlamaForCausalLM.from_pretrained(
            "/path/to/model",
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer #, model_config
