import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

from layers.mlp import MLP


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        # self.device = "cpu"
        print(self.device)

        self.llama = LlamaForCausalLM.from_pretrained(
            "./llama",
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
        )
        self.hidden_dim_of_llama = 4096
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        for name, param in self.llama.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (
                configs.use_multi_gpu and configs.local_rank == 0
            ):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)
            self.decoder = nn.Linear(self.hidden_dim_of_llama, 6)
        else:
            if not configs.use_multi_gpu or (
                configs.use_multi_gpu and configs.local_rank == 0
            ):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(
                self.token_len,
                self.hidden_dim_of_llama,
                configs.mlp_hidden_dim,
                configs.mlp_hidden_layers,
                configs.dropout,
                configs.mlp_activation,
            )
            self.decoder = MLP(
                self.hidden_dim_of_llama,
                6,
                configs.mlp_hidden_dim,
                configs.mlp_hidden_layers,
                configs.dropout,
                configs.mlp_activation,
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, seq_len, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(bs * n_vars, -1)

        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        # print("fold_out shape:", fold_out.shape)

        features_embeds = self.encoder(fold_out)

        outputs = self.llama.model(inputs_embeds=features_embeds)[0]
        dec_out = self.decoder(outputs)

        # Average dec_out
        dec_out = dec_out.mean(dim=1)
        current_batch = dec_out.shape[0] // n_vars
        dec_out = dec_out.view(current_batch, n_vars, 6).mean(dim=1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
