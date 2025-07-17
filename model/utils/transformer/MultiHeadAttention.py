import torch
import torch.nn as nn
import math



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, h: int, dropout: float) -> None:
        """
        Multi-Head Attention Block.

        Args:
            embed_dim (int): The dimension of the input embedding.
            h (int): The number of attention heads.
            dropout (float): Dropout rate for attention scores.
        """
        super().__init__()
        self.embed_dim = embed_dim  # Embedding vector size
        self.h = h  # Number of heads
        
        # Ensure the embedding dimension is divisible by the number of heads

        assert embed_dim % h == 0, "embed_dim must be divisible by h"
        
        self.d_k = embed_dim // h  # Dimension of each head's vector

        # Linear layers to project the input into Q, K, and V
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for query
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for key
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for value

        # Output linear layer
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for the final output

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, causal_mask=None, key_mask=None, dropout=None):
        """
        Compute scaled dot-product attention.

        Args:
            query (Tensor): (*prefix_dims, num_heads, query_len, d_k)
            key (Tensor): (*prefix_dims, num_heads, key_len, d_k)
            value (Tensor): (*prefix_dims, num_heads, key_len, d_k)
            causal_mask (Tensor, optional): Mask for causal (future) attention. 
                Shape: (*prefix_dims, query_len)
            key_mask (Tensor, optional): Mask to ignore specific keys (e.g., padding).
                Shape: (*prefix_dims, key_len)
            dropout (nn.Dropout, optional): Dropout to apply on attention weights.

        Returns:
            output (Tensor): (*prefix_dims, num_heads, query_len, d_k)
            attn_weights (Tensor): (*prefix_dims, num_heads, query_len, key_len)
        """
        d_k = query.shape[-1]  # Dimension of each head
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5) # (*prefix_dims, num_heads, query_seq_len, key_seq_len)

        # Apply key mask (e.g., for padding)
        if key_mask is not None:
            expand_key_mask = key_mask.unsqueeze(-2).unsqueeze(-2).expand(attention_scores.shape) # (*prefix_dims, h, query_seq_len, key_seq_len)
            attention_scores = attention_scores.masked_fill(expand_key_mask, -1e9)


        # Apply softmax to normalize the scores
        attention_scores = torch.softmax(attention_scores, dim=-1)

        # Apply causal mask (prevent future tokens)
        if causal_mask is not None:
            expand_causal_mask = causal_mask.unsqueeze(-1).unsqueeze(-1).expand(attention_scores.transpose(-3,-2).shape)  # (*prefix_dims, query_seq_len, h, key_seq_len)
            expand_causal_mask = expand_causal_mask.transpose(-3,-2)  # (*prefix_dims, h, query_seq_len, key_seq_len)
            attention_scores = attention_scores.masked_fill(expand_causal_mask, 0.0)

        # Apply dropout (if provided)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Compute the attention-weighted output
        return torch.matmul(attention_scores, value), attention_scores
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last embedding dimension into multiple heads.

        Input shape:
            (..., seq_len, embed_dim)
        
        Output shape:
            (..., num_heads, seq_len, d_k)

        Note:
            d_k = embed_dim // num_heads
        """
        *prefix_dims, seq_len, embed_dim = x.shape
        if torch.jit.is_tracing():
            pass  # Skip assertion during tracing
        else:
            assert embed_dim == self.h * self.d_k, "embed_dim must be divisible by number of heads"

        # Reshape: (..., seq_len, h, d_k)
        x = x.view(*prefix_dims, seq_len, self.h, self.d_k)

        # Permute: (..., h, seq_len, d_k)
        return x.permute(*range(len(prefix_dims)), -2, -3, -1).contiguous()

    def forward(self, q, k, v, causal_mask=None, key_mask=None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            q (Tensor): Query tensor of shape (*prefix_dims, seq_len, embed_dim)
            k (Tensor): Key tensor of shape (*prefix_dims, seq_len, embed_dim)
            v (Tensor): Value tensor of shape (*prefix_dims, seq_len, embed_dim)
            causal_mask (Tensor, optional): Causal mask of shape (*prefix_dims, seq_len)
            key_mask (Tensor, optional): Key mask of shape (*prefix_dims, key_len)

        Returns:
            Tensor: Output tensor of shape (*prefix_dims, seq_len, embed_dim)
        """
        # Linear projections for Q, K, V
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split the embeddings into h heads and reshape (batch_size, seq_len, embed_dim) --> (batch_size, h, seq_len, d_k)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # Compute attention
        x, self.attention_scores = self.attention(query, key, value, causal_mask, key_mask, self.dropout)

        # Combine heads: (*prefix_dims, num_heads, seq_len, d_k) → (*prefix_dims, seq_len, embed_dim)
        x = x.transpose(-3, -2).contiguous().view(*x.shape[:-3], -1, self.h * self.d_k)

        # Final linear transformation (batch_size, seq_len, embed_dim)
        return self.w_o(x)
    
    def get_attention_scores_mean(self):
        scores = self.attention_scores.transpose(1,2).contiguous() # (batch, query_seq_len, h, key_seq_len)
        mean = scores.mean(dim=2) # (batch, query_seq_len, key_seq_len)
        return mean