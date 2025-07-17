import torch
from tqdm import tqdm
import math

def calc_cosine_similarity(a: torch.Tensor, b: torch.Tensor, topk: int = -1):
    """
    Calculate cosine similarity between two tensors.
    
    Args:
        a (torch.Tensor): Tensor of shape (N, D).
        b (torch.Tensor): Tensor of shape (M, D).
        topk (int): If -1, returns the full similarity matrix of shape (N, M).
                    Otherwise, returns a tuple (topk_scores, topk_indices) where each has shape (N, topk).
                    
    Returns:
        torch.Tensor or tuple(torch.Tensor, torch.Tensor):
            If topk == -1:
                Returns cosine similarity matrix of shape (N, M).
            Otherwise:
                Returns a tuple of (topk_scores, topk_indices) where:
                    - topk_scores: Tensor of cosine similarity scores of shape (N, topk)
                    - topk_indices: Tensor of corresponding indices from tensor b of shape (N, topk)
    """
    # Normalize each vector in tensor a
    a_norm = a / a.norm(dim=1, keepdim=True)
    # Normalize each vector in tensor b
    b_norm = b / b.norm(dim=1, keepdim=True)
    
    # Compute the cosine similarity matrix (N, M)
    sim = a_norm @ b_norm.T
    
    if topk != -1:
        # For each vector in a, get the topk highest cosine similarities and their indices from tensor b
        topk_scores, topk_indices = torch.topk(sim, k=topk, dim=1)
        return topk_scores, topk_indices
    
    # Return the full cosine similarity matrix if topk is -1
    return sim



def calc_cosine_similarity_block(a: torch.Tensor, b: torch.Tensor, topk: int = -1,
                                 max_block_size_a: int = 1024, max_block_size_b: int = 1024,
                                 group_ids_a: torch.Tensor = None, group_ids_b: torch.Tensor = None, device: torch.device = torch.device('cpu')):
    """
    Calculate cosine similarity between two tensors using block processing to manage GPU memory.
    This function leverages calc_cosine_similarity to compute similarity for each block.
    Optionally, if group_ids_a and group_ids_b are provided, only entries where the group IDs match
    are retained in the ranking; non-matching entries are set to -1.
    
    Args:
        a (torch.Tensor): Tensor of shape (N, D).
        b (torch.Tensor): Tensor of shape (M, D).
        topk (int): If -1, returns the full similarity matrix of shape (N, M).
                    Otherwise, returns a tuple (topk_scores, topk_indices) with shape (N, topk).
        max_block_size_a (int): Maximum number of rows of tensor a to process at once.
        max_block_size_b (int): Maximum number of rows of tensor b to process at once.
        group_ids_a (torch.Tensor, optional): Tensor of shape (N,) containing group IDs for tensor a.
        group_ids_b (torch.Tensor, optional): Tensor of shape (M,) containing group IDs for tensor b.
                    
    Returns:
        torch.Tensor or tuple(torch.Tensor, torch.Tensor):
            If topk == -1:
                Returns cosine similarity matrix of shape (N, M) where entries with mismatched group IDs
                are set to -1.
            Otherwise:
                Returns a tuple (topk_scores, topk_indices), computed only from entries with matching group IDs.
                For rows with no candidate in the same group, the ranking will be filled with -1.
    """
    N = a.size(0)
    M = b.size(0)
    cpu_device = torch.device('cpu')
    
    total_blocks = math.ceil(N / max_block_size_a) * math.ceil(M / max_block_size_b)
    with tqdm(total=total_blocks, desc='Processing blocks', mininterval=0.5) as pbar:
        if topk == -1:
            # Allocate full similarity matrix on device
            sim_full = torch.empty((N, M), device=a.device)
            for i in range(0, N, max_block_size_a):
                a_block = a[i:i+max_block_size_a]
                for j in range(0, M, max_block_size_b):
                    b_block = b[j:j+max_block_size_b]
                    sim_block = calc_cosine_similarity(a_block, b_block, topk=-1)
                    if group_ids_a is not None and group_ids_b is not None:
                        group_a_block = group_ids_a[i:i+max_block_size_a]
                        group_b_block = group_ids_b[j:j+max_block_size_b]
                        group_mask = (group_a_block.unsqueeze(1) == group_b_block.unsqueeze(0))
                        sim_block = torch.where(group_mask, sim_block, torch.full_like(sim_block, -1.0))
                    sim_full[i:i+a_block.size(0), j:j+b_block.size(0)] = sim_block
                    pbar.update(1)
            return sim_full
        else:
            topk_scores_list = []
            topk_indices_list = []
            for i in range(0, N, max_block_size_a):
                a_block = a[i:i+max_block_size_a].to(device)
                block_best_scores = None
                block_best_indices = None
                for j in range(0, M, max_block_size_b):
                    b_block = b[j:j+max_block_size_b].to(device)
                    sim_block = calc_cosine_similarity(a_block, b_block, topk=-1)
                    if group_ids_a is not None and group_ids_b is not None:
                        group_a_block = group_ids_a[i:i+max_block_size_a].to(device)
                        group_b_block = group_ids_b[j:j+max_block_size_b].to(device)
                        group_mask = (group_a_block.unsqueeze(1) == group_b_block.unsqueeze(0))
                        sim_block = torch.where(group_mask, sim_block, torch.full_like(sim_block, -1.0))
                    current_topk = min(topk, sim_block.size(1))
                    block_topk_scores, block_topk_indices = torch.topk(sim_block, k=current_topk, dim=1)
                    # Adjust indices relative to tensor b
                    block_topk_indices = block_topk_indices + j
                    # Explicitly set indices to -1 where similarity is -1 (invalid group)
                    block_topk_indices = torch.where(block_topk_scores < -0.999999, 
                                                     torch.full_like(block_topk_indices, -1),
                                                     block_topk_indices)
                    if block_best_scores is None:
                        block_best_scores = block_topk_scores.to(cpu_device)
                        block_best_indices = block_topk_indices.to(cpu_device)
                    else:
                        merged_scores = torch.cat([block_best_scores, block_topk_scores.to(cpu_device)], dim=1)
                        merged_indices = torch.cat([block_best_indices, block_topk_indices.to(cpu_device)], dim=1)
                        block_best_scores, order = torch.topk(merged_scores, k=topk if topk < merged_scores.size(1) else merged_scores.size(1), dim=1)
                        block_best_indices = merged_indices.gather(1, order)
                    pbar.update(1)
                topk_scores_list.append(block_best_scores)
                topk_indices_list.append(block_best_indices)
            final_topk_scores = torch.cat(topk_scores_list, dim=0)
            final_topk_indices = torch.cat(topk_indices_list, dim=0)
            return final_topk_scores, final_topk_indices
        


def find_correct_position(topk_indices: torch.Tensor, correct_idx: torch.Tensor) -> torch.Tensor:
    """
    Returns the position (0-indexed) of the correct index in each row of topk_indices.
    If the correct index is not found in a row, returns -1 for that row.
    
    Args:
        topk_indices (torch.Tensor): Tensor of shape (N, topk) containing predicted indices.
        correct_idx (torch.Tensor): Tensor of shape (N,) containing the correct index for each sample.
    
    Returns:
        torch.Tensor: Tensor of shape (N,) where each element is the position of the correct index 
                      in the corresponding row of topk_indices (0-indexed), or -1 if not found.
    """
    N, k = topk_indices.shape
    # Create a boolean mask where each element is True if the value equals the correct index.
    mask = (topk_indices == correct_idx.unsqueeze(1))  # Shape: (N, topk)
    # Generate column indices for each row.
    col_indices = torch.arange(k, device=topk_indices.device).unsqueeze(0).expand(N, k)
    # For positions where the mask is False, assign a high value (k) so that min returns k if no match is found.
    pos_masked = torch.where(mask, col_indices, torch.full_like(col_indices, k))
    # Get the minimum index along each row; if no match, this will be k.
    first_occurrence = pos_masked.min(dim=1)[0]  # Shape: (N,)
    # Replace indices equal to k (i.e., no match) with -1.
    result = torch.where(first_occurrence == k, torch.full_like(first_occurrence, -1), first_occurrence)
    return result
    
if __name__ == '__main__':
    tensor_a_file = '/workspaces/Ms2z/mnt/data/workspace/main_20250218/trains/main_work_20250220_sotuken/output_1160k/latent_mona_chem_data/mean_tensor.pt'
    tensor_b_file = '/workspaces/Ms2z/mnt/data/workspace/main_20250218/trains/main_work_20250220_sotuken/output_1160k/latent_mona_ms_data/mean_tensor.pt'

    tensor_a = torch.load(tensor_a_file).to(torch.float32)
    tensor_b = torch.load(tensor_b_file).to(torch.float32)

    topk_scores, topk_indices = calc_cosine_similarity_block(tensor_a, tensor_b, topk=1000)
    pass