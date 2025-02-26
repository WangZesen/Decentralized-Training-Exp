import math
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from src.conf import Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 400):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerModule(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 num_heads: int,
                 num_layers: int,
                 d_feedforward: int,
                 pad_token_idx: int,
                 dropout: float = 0.3):
        super(TransformerModule, self).__init__()

        self._d_model = d_model
        self._pad_token_idx = pad_token_idx
        self._token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self._positional_encoding = PositionalEncoding(d_model)
        self._transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self._enc_dropout = nn.Dropout(dropout)
        self._dec_dropout = nn.Dropout(dropout)
        self._linear = nn.Linear(d_model, vocab_size, bias=False)
        self._linear.weight = self._token_embedding.weight
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self._token_embedding.weight.data.uniform_(-initrange, initrange)
    
    def generate_padding_mask(self, data: torch.Tensor) -> torch.Tensor:
        padding_mask = (data != self._pad_token_idx).float()
        padding_mask = padding_mask.masked_fill(padding_mask == 0, float('-inf'))
        padding_mask = padding_mask.masked_fill(padding_mask == 1, float(0.0))
        return padding_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_padding_mask = self.generate_padding_mask(src).detach()
        tgt_padding_mask = self.generate_padding_mask(tgt).detach()
        tgt_mask = self._transformer.generate_square_subsequent_mask(tgt.size(1), device=src.device, dtype=torch.float32).detach()

        src = self._token_embedding(src) * math.sqrt(self._d_model)
        tgt = self._token_embedding(tgt) * math.sqrt(self._d_model)
        src = self._positional_encoding(src)
        tgt = self._positional_encoding(tgt)
        src = self._enc_dropout(src)
        tgt = self._dec_dropout(tgt)

        output = self._transformer(src, tgt,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=src_padding_mask,
                                  tgt_is_causal=True,
                                  )
        
        output = self._linear(output)
        return output

    @torch.no_grad()
    def get_memory(self, src: torch.Tensor):
        src_padding_mask = self.generate_padding_mask(src).detach()
        src = self._token_embedding(src) * math.sqrt(self._d_model)
        src = self._positional_encoding(src)
        memory = self._transformer.encoder(src, src_key_padding_mask=src_padding_mask)
        return memory, src_padding_mask

    @torch.no_grad()
    def get_tgt_from_memory(self,
                            src_padding_mask: torch.Tensor,
                            memory: torch.Tensor,
                            tgt: torch.Tensor) -> torch.Tensor:
        tgt_padding_mask = self.generate_padding_mask(tgt).detach()
        tgt_mask = self._transformer.generate_square_subsequent_mask(tgt.size(1), device=memory.device, dtype=torch.float32)
        tgt = self._token_embedding(tgt) * math.sqrt(self._d_model)
        tgt = self._positional_encoding(tgt)
        output = self._transformer.decoder(tgt, memory,
                                           tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=tgt_padding_mask,
                                           memory_key_padding_mask=src_padding_mask,
                                           tgt_is_causal=True)
        output = self._linear(output)
        return output


def get_model(cfg: "Config", tokenizer: Tokenizer, token_pad: str):
    match cfg.train.model.arch.lower():
        case "transformer":
            return TransformerModule(
                vocab_size=tokenizer.get_vocab_size(),
                d_model=cfg.train.model.d_model,
                num_heads=cfg.train.model.num_heads,
                num_layers=cfg.train.model.num_layers,
                d_feedforward=cfg.train.model.dim_feedforward,
                pad_token_idx=tokenizer.token_to_id(token_pad),
                dropout=cfg.train.model.dropout
            )
        case _:
            raise ValueError(f"Unsupported architecture: {cfg.model.arch}")

