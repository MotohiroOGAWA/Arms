import torch


def create_expand_mask(mask, fill=True):
    """
    Expand a mask from [batch_size, seq_len] to [batch_size, seq_len, seq_len]
    and set positions where j > i (future positions) to False.

    Args:
        mask (torch.Tensor): Input mask of shape [batch_size, seq_len].

    Returns:
        torch.Tensor: Expanded mask of shape [batch_size, seq_len, seq_len].
    """
    batch_size, seq_len = mask.size()
    
    if fill:
        # Create future mask: [seq_len, seq_len], True where j <= i
        seq_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=mask.device))
    else:
        # Create future mask: [seq_len, seq_len], True where j == i
        seq_mask = torch.eye(seq_len, dtype=torch.bool, device=mask.device)
    
    # Expand input mask to [batch_size, seq_len, seq_len]
    expand_mask = mask.unsqueeze(-1).expand(-1, -1, seq_len)
    
    # Combine with the future mask
    expand_mask = ~expand_mask & seq_mask
    return ~expand_mask