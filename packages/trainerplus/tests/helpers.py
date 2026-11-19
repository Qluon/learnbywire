import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trainerplus import TokenDataset


def tokenizer():
    vocab = {word: i for i, word in enumerate(
        ['[PAD]', '[UNK]', '[EOS]', 'a', 'b', 'c', 'd', 'answer', 'question', 'user', 'assistant', 'ab'])}
    raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    result = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]',
                                    eos_token='[EOS]', unk_token='[UNK]')
    result.chat_template = "{% for message in messages %}{{ message['role'] + ' ' }}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] + ' ' + eos_token + ' ' }}{% endgeneration %}{% else %}{{ message['content'] + ' ' }}{% endif %}{% endfor %}"
    return result


def model(seed=91):
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=16, n_embd=8,
        n_layer=1, n_head=2, resid_pdrop=0.2, embd_pdrop=0.2, attn_pdrop=0.2,
        bos_token_id=2, eos_token_id=2, pad_token_id=0, use_cache=False))


def data(rows=8, dynamic=False):
    result = []
    for i in range(rows):
        ids = torch.tensor([3, 4, 5+i%4, 2, 0, 0][:4+i%3] if dynamic else [3, 4, 5+i%4, 2, 0, 0])
        mask = ids.ne(0)
        labels = ids.masked_fill(~mask, -100)
        result.append(dict(input_ids=ids, attention_mask=mask, labels=labels))
    return TokenDataset.from_rows(result)
