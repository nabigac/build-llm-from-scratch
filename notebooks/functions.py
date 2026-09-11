import torch
import torch.nn as nn
import tiktoken

from torch.utils.data import (
    DataLoader,
    Dataset,
)


class gpt_dataset_v1(Dataset):
    def __init__(
        self,
        text,
        tokenizer,
        max_length,
        stride,
    ):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(
            text,
            allowed_special={
                "<|endoftext|>"
            },
        )

        assert len(token_ids) > max_length

        for start_index in range(
            0,
            len(token_ids) - max_length,
            stride,
        ):
            input_chunk = token_ids[
                start_index:
                start_index + max_length
            ]

            target_chunk = token_ids[
                start_index + 1:
                start_index + max_length + 1
            ]

            self.input_ids.append(
                torch.tensor(
                    input_chunk,
                    dtype=torch.long,
                )
            )

            self.target_ids.append(
                torch.tensor(
                    target_chunk,
                    dtype=torch.long,
                )
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.target_ids[index],
        )


def create_dataloader_v1(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding(
        "gpt2"
    )

    dataset = gpt_dataset_v1(
        text=text,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return data_loader


class multi_head_attention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
    ):
        super().__init__()

        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )

        self.w_query = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias,
        )

        self.w_key = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias,
        )

        self.w_value = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias,
        )

        self.out_proj = nn.Linear(
            d_out,
            d_out,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    context_length,
                    context_length,
                ),
                diagonal=1,
            ),
        )

    def forward(self, x):
        (
            batch_size,
            num_tokens,
            d_in,
        ) = x.shape

        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        queries = queries.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        keys = keys.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        values = values.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = (
            queries
            @ keys.transpose(2, 3)
        )

        causal_mask = self.mask.bool()[
            :num_tokens,
            :num_tokens,
        ]

        attention_scores.masked_fill_(
            causal_mask,
            -torch.inf,
        )

        attention_weights = torch.softmax(
            attention_scores
            / self.head_dim ** 0.5,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        context_vectors = (
            attention_weights @ values
        )

        context_vectors = (
            context_vectors
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                num_tokens,
                self.d_out,
            )
        )

        return self.out_proj(
            context_vectors
        )


class layer_norm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5

        self.scale = nn.Parameter(
            torch.ones(emb_dim)
        )

        self.shift = nn.Parameter(
            torch.zeros(emb_dim)
        )

    def forward(self, x):
        mean = x.mean(
            dim=-1,
            keepdim=True,
        )

        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        normalized_x = (
            (x - mean)
            / torch.sqrt(
                variance + self.eps
            )
        )

        return (
            self.scale * normalized_x
            + self.shift
        )


class gelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        coefficient = torch.sqrt(
            torch.tensor(
                2.0 / torch.pi,
                device=x.device,
            )
        )

        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    coefficient
                    * (
                        x
                        + 0.044715
                        * torch.pow(x, 3)
                    )
                )
            )
        )


class feed_forward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                config["emb_dim"],
                4 * config["emb_dim"],
            ),
            gelu(),
            nn.Linear(
                4 * config["emb_dim"],
                config["emb_dim"],
            ),
        )

    def forward(self, x):
        return self.layers(x)


class transformer_block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = (
            multi_head_attention(
                d_in=config["emb_dim"],
                d_out=config["emb_dim"],
                context_length=config[
                    "context_length"
                ],
                dropout=config[
                    "drop_rate"
                ],
                num_heads=config[
                    "n_heads"
                ],
                qkv_bias=config[
                    "qkv_bias"
                ],
            )
        )

        self.feed_forward = (
            feed_forward(config)
        )

        self.norm_1 = layer_norm(
            config["emb_dim"]
        )

        self.norm_2 = layer_norm(
            config["emb_dim"]
        )

        self.shortcut_dropout = (
            nn.Dropout(
                config["drop_rate"]
            )
        )

    def forward(self, x):
        shortcut = x

        x = self.norm_1(x)
        x = self.attention(x)
        x = self.shortcut_dropout(x)
        x = x + shortcut

        shortcut = x

        x = self.norm_2(x)
        x = self.feed_forward(x)
        x = self.shortcut_dropout(x)
        x = x + shortcut

        return x


class gpt_model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = (
            nn.Embedding(
                config["vocab_size"],
                config["emb_dim"],
            )
        )

        self.position_embedding = (
            nn.Embedding(
                config["context_length"],
                config["emb_dim"],
            )
        )

        self.embedding_dropout = (
            nn.Dropout(
                config["drop_rate"]
            )
        )

        self.transformer_blocks = (
            nn.Sequential(
                *[
                    transformer_block(
                        config
                    )
                    for _ in range(
                        config["n_layers"]
                    )
                ]
            )
        )

        self.final_norm = layer_norm(
            config["emb_dim"]
        )

        self.output_head = nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False,
        )

    def forward(self, input_ids):
        (
            batch_size,
            sequence_length,
        ) = input_ids.shape

        token_embeddings = (
            self.token_embedding(
                input_ids
            )
        )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        x = (
            token_embeddings
            + position_embeddings
        )

        x = self.embedding_dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)

        logits = self.output_head(x)

        return logits


def generate_text_simple(
    model,
    token_ids,
    max_new_tokens,
    context_size,
):
    for _ in range(max_new_tokens):
        conditioned_token_ids = token_ids[
            :,
            -context_size:,
        ]

        with torch.no_grad():
            logits = model(
                conditioned_token_ids
            )

        last_position_logits = logits[
            :,
            -1,
            :,
        ]

        next_token_id = torch.argmax(
            last_position_logits,
            dim=-1,
            keepdim=True,
        )

        token_ids = torch.cat(
            (
                token_ids,
                next_token_id,
            ),
            dim=1,
        )

    return token_ids