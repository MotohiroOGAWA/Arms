import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_lib import LSTMGraph
from torch_geometric.data import Batch, Data
import math




class FragEmbeddings(nn.Module):
    STRICT_FP_THRESHOLD = 0.1
    def __init__(self, 
                 node_dim: int, edge_dim: int,
                 fp_subspace_dim: int,
                 attached_motif_index_map: torch.Tensor, # (motif_size, attach_size)
                 bonding_cnt_tensor: torch.Tensor, # (max(attached_motif_index_map))
                 atom_layer_list: list,
                 lstm_iterations: int,
                 motif_fp_tensor: torch.Tensor,
                 attached_motif_fp_tensor: torch.Tensor,
                 connect_to_fp_tensor: torch.Tensor,
                 fp_dropout: float,
                 bos, pad, unk,
                 ) -> None:
        super().__init__()
        attached_motif_size = torch.max(attached_motif_index_map)+1
        assert attached_motif_size == bonding_cnt_tensor.size(0), "attached_motif_size and bonding_cnt_tensor size mismatch"
        assert attached_motif_size == len(atom_layer_list), "attached_motif_size and attached_motif_index_map size mismatch"

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.fp_subspace_dim = fp_subspace_dim
        self.motif_size = attached_motif_index_map.size(0)
        self.att_size = attached_motif_index_map.size(1)
        self.attached_motif_size = attached_motif_size
        self.attached_motif_index_map = nn.Parameter(attached_motif_index_map, requires_grad=False)
        self.bonding_cnt_tensor = nn.Parameter(bonding_cnt_tensor, requires_grad=False) # (max(attached_motif_index_map))
        self.max_bonding_cnt = torch.max(bonding_cnt_tensor)
        # self.atom_layer_list = atom_layer_list

        # attached motif id to motif id and attachment id
        row_indices, col_indices = torch.meshgrid(
            torch.arange(self.motif_size, dtype=torch.long),
            torch.arange(self.att_size, dtype=torch.long),
            indexing="ij"
        )
        row_col_pairs = torch.stack([row_indices.flatten(), col_indices.flatten()], dim=1)
        valid_mask = attached_motif_index_map.flatten() != -1
        attached_motif_idx_to_sep = row_col_pairs[valid_mask]
        self.attached_motif_idx_to_sep = nn.Parameter(attached_motif_idx_to_sep, requires_grad=False)

        atom_node_dim = atom_layer_list[3]['x'].size(1)
        atom_edge_attr_dim = atom_layer_list[3]['edge_attr'].size(1)
        for i, atom_layer in enumerate(atom_layer_list):
            if i < 3:
                continue
            nodes = atom_layer['x']
            edges = atom_layer['edge_index']
            edges_attr = atom_layer['edge_attr']
            assert nodes.size(1) == atom_node_dim, "atom_node_dim mismatch"
            if edges_attr.size(0) > 0:
                assert edges_attr.size(1) == atom_edge_attr_dim, "atom_edge_attr_dim mismatch"
            else:
                edges = torch.zeros(2, 1, dtype=edges.dtype)
                edges_attr = torch.zeros(1, atom_edge_attr_dim, dtype=edges_attr.dtype)

            self.register_buffer(f'atom_layer_x_{i}', nodes)
            self.register_buffer(f'atom_layer_edge_index_{i}', edges)
            self.register_buffer(f'atom_layer_edge_attr_{i}', edges_attr)
        self.atom_node_dim = atom_node_dim
        self.atom_edge_attr_dim = atom_edge_attr_dim

        self.bos = bos
        self.pad = pad
        self.unk = unk
        self.special_token_embedding = nn.Embedding(3, node_dim)

        self.fp_dim = motif_fp_tensor.size(1)
        self.motif_fp_tensor = nn.Parameter(motif_fp_tensor, requires_grad=False) # (motif_size, fp_size)
        self.attached_motif_fp_tensor = nn.Parameter(attached_motif_fp_tensor, requires_grad=False) # (attached_motif_size, fp_size)
        self.connect_to_fp_tensor = nn.Parameter(connect_to_fp_tensor, requires_grad=False) # (attached_motif_size, max_bonding_cnt, fp_size)
        self.attached_motif_embedding_layer = nn.Embedding(self.attached_motif_size, node_dim)
        self.atom_layer = LSTMGraph(
            node_dim=self.atom_node_dim, edge_dim=self.atom_edge_attr_dim,
            node_h_size=node_dim, edge_h_size=edge_dim,
            iterations=lstm_iterations, directed=False,
        )
        self.latent_atom_layer_to_motif_fp = nn.Sequential(
            nn.Linear(fp_subspace_dim, motif_fp_tensor.size(1)*2),
            nn.ReLU(),
            nn.Linear(motif_fp_tensor.size(1)*2, motif_fp_tensor.size(1)),
        )
        self.latent_atom_layer_to_attached_motif_fp = nn.Sequential(
            nn.Linear(fp_subspace_dim, attached_motif_fp_tensor.size(1)*2),
            nn.ReLU(),
            nn.Linear(attached_motif_fp_tensor.size(1)*2, attached_motif_fp_tensor.size(1)),
        )
        self.latent_atom_layer_to_connect_fp = nn.Sequential(
            nn.Linear(fp_subspace_dim+edge_dim, connect_to_fp_tensor.size(2)*2),
            nn.ReLU(),
            nn.Linear(connect_to_fp_tensor.size(2)*2, connect_to_fp_tensor.size(2)),
        )
        self.connect_linear = nn.Linear(self.max_bonding_cnt, edge_dim)
        self.base_connect_tensor = nn.Parameter(torch.full((self.max_bonding_cnt,), -1.0, dtype=torch.float32), requires_grad=False)

        self.fp_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.fp_dropout = nn.Dropout(fp_dropout)

        self.edge_linear = nn.Sequential(
            nn.Linear(self.max_bonding_cnt, edge_dim),
        )

        self.calc_attached_motif_idx_to_embeddings = {}
        
    def forward(self, idx):
        if idx.size(-1) == 2: 
            # (..., 2) --> (..., node_dim)
            return self.embed_attached_motif(idx)
        elif idx.size(-1) == 3:
            # (..., 3) --> (..., node_dim + edge_dim)
            node_embeddings = self.embed_attached_motif(idx[..., :2])
            edge_attr = self.embed_edge_attr(idx)
            return torch.cat([node_embeddings, edge_attr], dim=-1)
        else:
            raise ValueError(f'Invalid input shape: {idx.size()}')
        
    def reset_calc_embeddings(self):
        self.calc_attached_motif_idx_to_embeddings = {}

    def embed_attached_motif(self, idx):
        # (..., 2) --> (..., node_dim)
        original_shape = idx.shape[:-1]  
        flattened_idx = idx.reshape(-1, 2)  # (N, 2)

        # Mask to separate special tokens (0~2) and normal indices
        special_token_mask = flattened_idx[:, 0] <= 2
        normal_token_mask = ~special_token_mask

        # Handle special tokens
        special_tokens = flattened_idx[special_token_mask, 0]  # Extract first column for special tokens
        special_embeddings = self.special_token_embedding(special_tokens)  # (num_special_tokens, node_dim)

        full_embeddings = torch.zeros(
            (flattened_idx.size(0), self.node_dim),  # (N, node_dim)
            device=idx.device
        )

        # Assign special and normal embeddings
        full_embeddings[special_token_mask] = special_embeddings
        full_embeddings[normal_token_mask] = self.attached_motif_embedding_layer(self.attached_motif_index_map[flattened_idx[normal_token_mask][:,0], flattened_idx[normal_token_mask][:,1]])

        return full_embeddings.view(*original_shape, -1)


    def embed_attached_motif_from_atom_layer(self, idx, unique=True):
        # (..., 2) --> (..., node_dim)
        original_shape = idx.shape[:-1]  
        flattened_idx = idx.reshape(-1, 2)  # (N, 2)

        # Mask to separate special tokens (0~2) and normal indices
        special_token_mask = flattened_idx[:, 0] <= 2
        normal_token_mask = ~special_token_mask

        # Handle special tokens
        special_tokens = flattened_idx[special_token_mask, 0]  # Extract first column for special tokens
        special_embeddings = self.special_token_embedding(special_tokens)  # (num_special_tokens, node_dim)

        if unique:
            # Handle normal tokens
            unique_idx, inverse_indices = torch.unique(flattened_idx[normal_token_mask], dim=0, return_inverse=True)

            cached_embeddings = []
            new_indices = []
            for i, idx_tuple in enumerate(unique_idx.tolist()):
                idx_tuple = tuple(idx_tuple)
                if idx_tuple in self.calc_attached_motif_idx_to_embeddings:
                    cached_embeddings.append(self.calc_attached_motif_idx_to_embeddings[idx_tuple])
                else:
                    new_indices.append(i)

            if new_indices:
                new_unique_idx = unique_idx[new_indices]
                new_embeddings = self.embed_atom_layer(new_unique_idx)  # (new_size, node_dim)

                for i, idx_tuple in zip(new_indices, new_unique_idx.tolist()):
                    self.calc_attached_motif_idx_to_embeddings[tuple(idx_tuple)] = new_embeddings[i]

                cached_embeddings.extend(new_embeddings)
            normal_embeddings = torch.stack(cached_embeddings, dim=0)  # (unique_datasize, node_dim)
        else:
            normal_embeddings = self.embed_atom_layer(flattened_idx[normal_token_mask])



        # Create full embedding tensor
        full_embeddings = torch.zeros(
            (flattened_idx.size(0), self.node_dim),  # (N, node_dim)
            device=idx.device
        )

        # Assign special and normal embeddings
        full_embeddings[special_token_mask] = special_embeddings
        if unique:
            full_embeddings[normal_token_mask] = normal_embeddings[inverse_indices]
        else:
            full_embeddings[normal_token_mask] = normal_embeddings

        # Reshape back to original shape (..., node_dim)
        final_embeddings = full_embeddings.view(*original_shape, -1)

        return final_embeddings
    
    def embed_edge_attr(self, bond_pos_tensor):
        """
        Convert bond position tensor to one-hot encoded edge attributes.

        Args:
            bond_pos_tensor (torch.Tensor): Tensor of shape (datasize, 3), where each row contains
                                            (motif_idx, attachment_idx, bond_pos).

        Returns:
            torch.Tensor: One-hot encoded edge attributes of shape (datasize, max_bonding_cnt).
        """
        # Extract motif_idx, attach_idx, and bond_pos
        motif_idx = bond_pos_tensor[..., 0]
        attach_idx = bond_pos_tensor[..., 1]
        bond_pos = bond_pos_tensor[..., 2]

        # Get bonding count for each pair (motif_idx, attach_idx)
        indices = self.attached_motif_index_map[motif_idx, attach_idx]
        bond_cnt = self.bonding_cnt_tensor[indices]  # Shape: (datasize,)

        # Create a full tensor filled with -1.0
        broadcast_shape = (*bond_pos_tensor.shape[:-1], self.max_bonding_cnt.item())  # (..., max_bonding_cnt)
        one_hot_tensor = torch.full(
            broadcast_shape,
            -1.0,
            dtype=torch.float32,
            device=bond_pos_tensor.device
        )

        # Generate index grid for broadcasting
        bond_idx_range = torch.arange(self.max_bonding_cnt, device=bond_pos_tensor.device).view(
            *([1] * (bond_pos_tensor.ndim - 1)), self.max_bonding_cnt
        )  # Shape: (..., max_bonding_cnt)

        # Create mask for valid positions where bond_cnt is greater than the index
        mask = bond_idx_range < bond_cnt.unsqueeze(-1)  # Shape: (..., max_bonding_cnt)

        # Fill valid positions with 0.0
        one_hot_tensor[mask] = 0.0

        # Convert bond_pos to indexing format and set bond positions to 1.0
        bond_pos_expanded = bond_pos.unsqueeze(-1).expand_as(one_hot_tensor)
        bond_mask = bond_idx_range == bond_pos_expanded
        one_hot_tensor[bond_mask] = 1.0

        edge_attr = self.edge_linear(one_hot_tensor) # (datasize, edge_dim)

        return edge_attr
    
    def get_connet_tensor(self, attached_motif_idx):
        bonding_cnt = self.bonding_cnt_tensor[attached_motif_idx] # (batch_size,)

        base_connect_ex_tensor = self.base_connect_tensor.clone().expand(len(attached_motif_idx), self.max_bonding_cnt)

        valid_mask = torch.arange(self.max_bonding_cnt, device=bonding_cnt.device).unsqueeze(0) < bonding_cnt.unsqueeze(1)

        connect_tensor = base_connect_ex_tensor.masked_fill(valid_mask, 0.0)

        eye_matrix = torch.diag_embed(valid_mask).to(torch.float32)  # (batch_size, max_bonding_cnt, max_bonding_cnt)
        # eye_matrix = torch.eye(self.max_bonding_cnt, device=connect_tensor.device).unsqueeze(0)  # (1, max_bonding_cnt, max_bonding_cnt)
        connect_tensor = connect_tensor.unsqueeze(1) + eye_matrix  # (batch_size, max_bonding_cnt, max_bonding_cnt)

        return connect_tensor, valid_mask
    
    def get_connect_flat_tensor(self, attached_motif_idx):
        """
        Creates a connection matrix (connect_tensor) based on attached_motif_idx and applies a linear transformation.

        Args:
            attached_motif_idx (Tensor): A tensor of shape [batch_size], containing the indices of attached motifs
                                        to be processed in each batch.

        Returns:
            connect_tensor_flat (Tensor): A tensor of shape [num_valid_entries, max_bonding_cnt].
                                        This is the masked connection matrix after applying a linear transformation.
            mask (Tensor): A boolean tensor of shape [batch_size, max_bonding_cnt].
                        Indicates valid bonding locations for each batch.
            valid_batch_ids (Tensor): A tensor of shape [num_valid_entries].
                                    Batch indices where `mask` is `True`.
            valid_bond_ids (Tensor): A tensor of shape [num_valid_entries].
                                    Bonding indices where `mask` is `True`.
        """
        connect_tensor, v_mask = self.get_connet_tensor(attached_motif_idx)
        mask_flat = v_mask.reshape(-1)
        valid_mask = v_mask.nonzero(as_tuple=True)
        valid_batch_ids, valid_bond_ids = valid_mask

        eye_matrix = torch.eye(self.max_bonding_cnt, device=connect_tensor.device).unsqueeze(0)  # (1, max_bonding_cnt, max_bonding_cnt)
        connect_tensor = connect_tensor.unsqueeze(1) + eye_matrix  # (batch_size, max_bonding_cnt, max_bonding_cnt)
        connect_tensor_flat = connect_tensor.reshape(-1, self.max_bonding_cnt)[mask_flat]  # (batch_size * max_bonding_cnt, max_bonding_cnt)

        return connect_tensor_flat, v_mask, valid_batch_ids, valid_bond_ids


    def get_atom_layer(self, idx):
        """
        Retrieve a Data object from the registered buffers.

        Args:
            idx (int): Index of the desired graph.

        Returns:
            Data: The corresponding Data object reconstructed from buffers.
        """
        x = getattr(self, f"atom_layer_x_{idx}")
        edge_index = getattr(self, f"atom_layer_edge_index_{idx}")
        edge_attr = getattr(self, f"atom_layer_edge_attr_{idx}")
        # if edge_attr.numel() == 0:  # Handle empty edge_attr case
        #     edge_attr = None
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def batched_atom_layer(self, idx_tensor):
        selected_graphs = [self.get_atom_layer(self.attached_motif_index_map[idx[0],idx[1]]) for idx in idx_tensor]
        batched_data = Batch.from_data_list(selected_graphs)
        return batched_data
    
    def embed_atom_layer(self, idx_tensor):
        """
        Embed a batch of atom layers.
        """
        batched_atom_layer = self.batched_atom_layer(idx_tensor)
        x = self.atom_layer(batched_atom_layer)
        return x # (batchsize, node_dim)
    
    def calc_motif_fp_loss(self, embed, motif_idx):
        embed = embed[..., :self.fp_subspace_dim]
        predicted_fp = self.latent_atom_layer_to_motif_fp(self.fp_dropout(embed)) # (..., fp_size)
        target_fp = self.motif_fp_tensor[motif_idx]

        fp_loss, fp_acc, strict_fp_acc \
            = self.calc_fp_loss(predicted_fp, target_fp, strict_threshold=FragEmbeddings.STRICT_FP_THRESHOLD)
        return fp_loss, fp_acc, strict_fp_acc
    
    def calc_attached_motif_fp_loss(self, embed, attached_motif_idx):
        embed = embed[..., :self.fp_subspace_dim]
        predicted_fp = self.latent_atom_layer_to_attached_motif_fp(self.fp_dropout(embed)) # (..., fp_size)
        target_fp = self.attached_motif_fp_tensor[attached_motif_idx]

        fp_loss, fp_acc, strict_fp_acc \
            = self.calc_fp_loss(predicted_fp, target_fp, strict_threshold=FragEmbeddings.STRICT_FP_THRESHOLD)
        return fp_loss, fp_acc, strict_fp_acc
    
    def calc_connect_fp_loss(self, embed, attached_motif_idx):
        embed = embed[..., :self.fp_subspace_dim]
        connect_tensor_flat, mask, valid_batch_ids, valid_bond_ids = self.get_connect_flat_tensor(attached_motif_idx)
        # bonding_cnt = self.bonding_cnt_tensor[attached_motif_idx] # (batch_size,)

        # base_connect_ex_tensor = self.base_connect_tensor.clone().expand(len(attached_motif_idx), self.max_bonding_cnt)

        # mask = torch.arange(self.max_bonding_cnt, device=bonding_cnt.device).unsqueeze(0) < bonding_cnt.unsqueeze(1)
        # mask_flat = mask.reshape(-1)
        # valid_mask = mask.nonzero(as_tuple=True)
        # valid_batch_ids, valid_bond_ids = valid_mask

        # connect_tensor = base_connect_ex_tensor.masked_fill(mask, 0.0)

        # eye_matrix = torch.eye(self.max_bonding_cnt, device=connect_tensor.device).unsqueeze(0)  # (1, max_bonding_cnt, max_bonding_cnt)
        # connect_tensor = connect_tensor.unsqueeze(1) + eye_matrix  # (batch_size, max_bonding_cnt, max_bonding_cnt)
        # connect_tensor_flat = connect_tensor.reshape(-1, self.max_bonding_cnt)[mask_flat]  # (batch_size * max_bonding_cnt, max_bonding_cnt)

        embed_ex = embed.unsqueeze(1).repeat(1, self.max_bonding_cnt, 1)
        embed_flat = embed_ex.reshape(-1, embed.size(-1))[mask.reshape(-1)]
        # embed_flat = embed_ex.reshape(-1, embed.size(-1))[mask_flat]

        connect_tensor_flat = self.connect_linear(connect_tensor_flat)
        x = torch.cat([embed_flat, connect_tensor_flat], dim=-1) # (flat_size, node_dim+edge_dim)
        x = self.latent_atom_layer_to_connect_fp(self.fp_dropout(x))  # (flat_size, fp_size)
        
        fp_tensor = self.connect_to_fp_tensor[attached_motif_idx[valid_batch_ids], valid_bond_ids]
        fp_loss, fp_acc, strict_fp_acc = self.calc_fp_loss(x, fp_tensor, strict_threshold=FragEmbeddings.STRICT_FP_THRESHOLD)
        return fp_loss, fp_acc, strict_fp_acc


    def train_atom_layer_all(self, batch_size, shuffle=True):
        # Get all token ids
        token_ids = torch.arange(3, self.attached_motif_size, device=self.attached_motif_index_map.device)

        # Shuffle token_ids if shuffle=True
        if shuffle:
            token_ids = token_ids[torch.randperm(token_ids.size(0), device=token_ids.device)]

        # Process in batches
        for i in range(0, token_ids.size(0), batch_size):
            # Get the current batch of token_ids
            batch_token_ids = token_ids[i:i+batch_size]
            
            # Train the model using the current batch
            yield self.train_atom_layer(batch_token_ids)
    
    def train_atom_layer(self, attached_motif_indices):
        """
        Train the atom layer for the given batch of indices.
        """
        embed_graph = self.embed_atom_layer(self.attached_motif_idx_to_sep[attached_motif_indices])

        motif_fp_res, attached_motif_fp_res, connect_fp_res = self.calc_embed_fp_loss(embed_graph, attached_motif_indices)
        
        return motif_fp_res, attached_motif_fp_res, connect_fp_res
    
    def calc_embed_fp_loss(self, embed, attached_motif_indices):

        motif_indices = self.attached_motif_idx_to_sep[attached_motif_indices][:,0]
        fp_loss, fp_acc, strict_fp_acc = self.calc_motif_fp_loss(embed, motif_indices)
        motif_fp_res = (fp_loss, fp_acc, strict_fp_acc)

        fp_loss, fp_acc, strict_fp_acc = self.calc_attached_motif_fp_loss(embed, attached_motif_indices)
        attached_motif_fp_res = (fp_loss, fp_acc, strict_fp_acc)

        fp_loss, fp_acc, strict_fp_acc = self.calc_connect_fp_loss(embed, attached_motif_indices)
        connect_fp_res = (fp_loss, fp_acc, strict_fp_acc)

        return motif_fp_res, attached_motif_fp_res, connect_fp_res


    def calc_fp_loss(self, predicted_fp, target_fp, strict_threshold):
        """
        Compute the loss and a stricter accuracy for fingerprint prediction.

        Args:
            predicted_fp (torch.Tensor): Model's predicted fingerprint tensor (batch_size, fp_size).
            target_fp (torch.Tensor): Ground truth fingerprint tensor (batch_size, fp_size).
            strict_threshold (float): Threshold for strict accuracy (default: 0.1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing fp_loss, fp_acc, and strict_fp_acc.
        """
        # Compute the loss for the current batch
        element_wise_loss  = self.fp_loss_fn(predicted_fp, target_fp)

        # Create masks for 0 and 1 in fp_tensor
        zero_mask = (target_fp == 0)  # Shape: [batch_size, fp_size]
        one_mask = (target_fp == 1)  # Shape: [batch_size, fp_size]

        # Compute the mean loss for zero_mask and one_mask
        zero_loss_mean = torch.sum(element_wise_loss * zero_mask, dim=1) / (zero_mask.sum(dim=1) + 1e-8)  # Shape: [batch_size]
        one_loss_mean = torch.sum(element_wise_loss * one_mask, dim=1) / (one_mask.sum(dim=1) + 1e-8)    # Shape: [batch_size]

        # Compute the final loss as the average of zero_loss_mean and one_loss_mean
        fp_loss = (zero_loss_mean + one_loss_mean).mean() / 2

        
        predicted_fp = torch.sigmoid(predicted_fp)

        # Compute accuracy: (number of correct predictions) / (total elements)
        rounded_pred_fp = torch.round(torch.clamp(predicted_fp, min=0.0, max=1.0))
        correct_predictions = (rounded_pred_fp == target_fp).float()

        # Compute the mean loss for zero_mask and one_mask
        zero_acc_mean = torch.sum(correct_predictions * zero_mask, dim=1) / (zero_mask.sum(dim=1) + 1e-8)  # Shape: [batch_size]
        one_acc_mean = torch.sum(correct_predictions * one_mask, dim=1) / (one_mask.sum(dim=1) + 1e-8)    # Shape: [batch_size]

        # Compute the final accuracy as the average of zero_acc_mean and one_acc_mean
        fp_acc = (zero_acc_mean + one_acc_mean).mean() / 2

        # **Stricter accuracy computation**
        # Create strict rounding conditions based on the `strict_threshold`
        rounded_strict_fp = torch.full_like(predicted_fp, -1.0)  # Initialize all as incorrect (-1.0)

        # Values close to 0 are considered 0
        rounded_strict_fp[(predicted_fp >= -strict_threshold) & (predicted_fp <= strict_threshold)] = 0.0
        # Values close to 1 are considered 1
        rounded_strict_fp[(predicted_fp >= (1.0 - strict_threshold)) & (predicted_fp <= (1.0 + strict_threshold))] = 1.0

        # Compute strict accuracy: number of exact matches
        strict_correct_predictions = (rounded_strict_fp == target_fp).float()

        # Compute the mean accuracy for zero_mask and one_mask using strict criteria
        strict_zero_acc_mean = torch.sum(strict_correct_predictions * zero_mask, dim=1) / (zero_mask.sum(dim=1) + 1e-8)  # Shape: [batch_size]
        strict_one_acc_mean = torch.sum(strict_correct_predictions * one_mask, dim=1) / (one_mask.sum(dim=1) + 1e-8)  # Shape: [batch_size]

        # Compute the final strict accuracy as the average of strict_zero_acc_mean and strict_one_acc_mean
        strict_fp_acc = (strict_zero_acc_mean + strict_one_acc_mean).mean() / 2

        return fp_loss, fp_acc, strict_fp_acc


    def initialize_atom_layer_from_pretrained(self,batch_size=None):
        if batch_size is None:
            batch_size = self.attached_motif_size
        init_tensor = torch.zeros((self.attached_motif_size, self.node_dim), dtype=torch.float32, device=self.attached_motif_index_map.device)
        token_ids = torch.arange(3, self.attached_motif_size, device=self.attached_motif_index_map.device)
        
        for i in range(0, token_ids.size(0), batch_size):
            # Get the current batch of token_ids
            batch_token_ids = token_ids[i:i+batch_size]

            embed_graph = self.embed_atom_layer(self.attached_motif_idx_to_sep[batch_token_ids])
            
            init_tensor[batch_token_ids] = embed_graph

        self.attached_motif_embedding_layer.weight.data.copy_(init_tensor)

    def load_atom_layer_from_weight(self, file):
        loaded_weights = torch.load(file)
        self.attached_motif_embedding_layer.weight.data.copy_(loaded_weights)

    def save_atom_layer_weight(self, file):
        torch.save(self.attached_motif_embedding_layer.weight.data, file)

    def freeze_latent_to_fp_layers(self):
        """
        Freeze the weights of specific layers to prevent them from being updated during training.
        """
        layers_to_freeze = [
            self.latent_atom_layer_to_motif_fp,
            self.latent_atom_layer_to_attached_motif_fp,
            self.latent_atom_layer_to_connect_fp
        ]
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False  # Disable gradient updates



class MsEmbeddings(nn.Module):
    def __init__(self, precursor_dim, precursor_hidden_dims: list[int], fragment_dim, fragment_hidden_dims: list[int], embed_dim: int, dropout=0.1) -> None:
        super().__init__()
        self.precursor_dim = precursor_dim
        self.fragment_dim = fragment_dim
        self.embed_dim = embed_dim

        # Precursor fully connected layers (MLP)
        precursor_layers = []
        prev_dim = precursor_dim
        for hidden_dim in precursor_hidden_dims:
            precursor_layers.append(nn.Linear(prev_dim, hidden_dim))
            precursor_layers.append(nn.ReLU())
            precursor_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        precursor_layers.append(nn.Linear(prev_dim, embed_dim))
        self.precursor_embeddings = nn.Sequential(*precursor_layers)

        # Fragment fully connected layers (MLP)
        fragment_layers = []
        prev_dim = fragment_dim
        for hidden_dim in fragment_hidden_dims:
            fragment_layers.append(nn.Linear(prev_dim, hidden_dim))
            fragment_layers.append(nn.ReLU())
            fragment_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        fragment_layers.append(nn.Linear(prev_dim, embed_dim))
        self.fragment_embeddings = nn.Sequential(*fragment_layers)

        
    def forward(self, x, is_precursor: bool):
        if is_precursor:
            x = self.precursor_embeddings(x)
        else:
            x = self.fragment_embeddings(x)
        return x
        
class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, embed_dim)
        return self.dropout(self.embedding(x))

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 6000, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # Make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # Add constant positional encoding to the embedding
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len].detach()  # detach() で計算グラフから切り離す

        if x.is_cuda:  # GPU に転送する場合
            pe = pe.cuda()
            
        x = x + pe  # 位置エンコーディングを加算
        x = self.dropout(x)  # Dropout を適用
        return x

class FeedForwardBlock(nn.Module):

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, ff_dim) # w1 and b1
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, embed_dim) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, embed_dim) --> (batch, seq_len, d_ff) --> (batch, seq_len, embed_dim)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear_2(x)
    


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, edge_dim, directed=False, softmax_per="dst", T=1.0):
        """
        Graph Attention Layer (GAT) with batched graphs and optimized node indexing.

        Args:
            in_features (int): Number of input node features.
            out_features (int): Number of output node features.
            edge_dim (int): Dimension of edge attributes.
            directed (bool): Whether the graph is directed or not.
            softmax_per (str): Whether to apply softmax "graph"-wise or "dst"-wise.
                - "graph": Normalizes attention scores across the entire graph.
                - "dst": Normalizes attention scores per destination node.
            T (float): Temperature parameter for softmax scaling.
                - Controls the sharpness of the attention distribution.
                - Higher `T` (>1.0) makes attention scores more uniform (soft attention).
                - Lower `T` (<1.0) makes attention scores sharper, emphasizing higher weights.
                - `T=1.0` applies standard softmax without additional scaling.
        """
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))  # Weight matrix
        self.a = nn.Parameter(torch.Tensor(2 * out_features + edge_dim, 1))  # Attention mechanism
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.directed = directed  # Whether the graph is directed or not

        assert softmax_per in ["graph", "dst"], "Invalid softmax_per value. Choose from ['graph', 'dst']"
        self.softmax_per = softmax_per # Softmax per graph or per destination node

        self.T = T  # Temperature for softmax

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the learnable parameters using Xavier initialization.
        """
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index, edge_attr, node_mask, edge_mask):
        """
        Forward pass for the GAT layer.

        Args:
            x (torch.Tensor): Node feature matrix of shape [batch_size, seq_len, in_features].
            edge_index (torch.Tensor): Graph edge connections of shape [batch_size, num_edges, 2].
            edge_attr (torch.Tensor): Edge attributes of shape [batch_size, num_edges, edge_dim].
            node_mask (torch.Tensor): Boolean mask for invalid nodes [batch_size, seq_len] (True = invalid).
            edge_mask (torch.Tensor): Boolean mask for invalid edges [batch_size, num_edges] (True = invalid).

        Returns:
            torch.Tensor: Updated node embeddings of shape [batch_size, seq_len, out_features].
        """
        batch_size, seq_len, node_dim = x.shape
        _, num_edges, edge_dim = edge_attr.shape

        # **1. Extract valid nodes (where node_mask is False)**
        node_valid_indices = (~node_mask).nonzero(as_tuple=True)  # Shape: [valid_nodes_count, 2]
        node_batch_ids = node_valid_indices[0]  # Batch indices of valid nodes
        valid_node_ids = node_valid_indices[1]  # Node indices within each batch

        # **2. Compute the cumulative offset for valid nodes in each batch**
        valid_counts_per_batch = (~node_mask).sum(dim=1)  # Shape: [batch_size]
        batch_offsets = torch.cumsum(valid_counts_per_batch, dim=0)  # Compute cumulative sum
        batch_offsets = torch.cat([torch.tensor([0], device=x.device), batch_offsets[:-1]])  # Shift for indexing

        # **3. Assign unique global node IDs based on valid nodes only**
        global_node_ids = torch.arange(len(valid_node_ids), device=x.device)

        global_to_batch_node_map = torch.stack([node_batch_ids, valid_node_ids], dim=1)
        batch_node_to_global_map = -torch.ones((batch_size, seq_len), dtype=torch.long, device=x.device)
        batch_node_to_global_map[node_batch_ids, valid_node_ids] = global_node_ids  


        edge_valid_indices = (~edge_mask).nonzero(as_tuple=True)  # Shape: [valid_edges_count, 2]
        edge_batch_ids = edge_valid_indices[0]
        valid_edge_ids = edge_valid_indices[1]

        edge_index_flat = edge_index[edge_batch_ids, valid_edge_ids]  # Shape: [valid_edges_count, 2]
        edge_attr_flat = edge_attr[edge_batch_ids, valid_edge_ids]  # Shape: [valid_edges_count, edge_dim]
        
        src = edge_index_flat[:, 0]
        src = batch_node_to_global_map[edge_batch_ids, src]
        dst = edge_index_flat[:, 1]
        dst = batch_node_to_global_map[edge_batch_ids, dst]

        if not self.directed:
            edge_batch_ids = torch.cat([edge_batch_ids, edge_batch_ids], dim=0)
            edge_attr_flat = torch.cat([edge_attr_flat, edge_attr_flat], dim=0)

            # Add reverse edges for undirected graphs
            src, dst = torch.cat([src, dst]), torch.cat([dst, src])
        
        src = torch.cat([src, global_node_ids], dim=0)
        dst = torch.cat([dst, global_node_ids], dim=0)
        edge_batch_ids = torch.cat([edge_batch_ids, node_batch_ids], dim=0)
        empty_edge_attr = torch.zeros(len(global_node_ids), edge_dim, dtype=edge_attr.dtype, device=edge_attr.device)
        edge_attr_flat = torch.cat([edge_attr_flat, empty_edge_attr], dim=0)


        x = torch.matmul(x, self.W)  # Shape: [batch_size, seq_len, out_features]
        x_flat = x.reshape(-1, x.shape[-1])[(~node_mask).reshape(-1)] # Shape: [valid_nodes_count, out_features]
        x_transformed = x_flat[global_node_ids]  # Only use valid nodes

        h_src = x_transformed[src]  # Shape: [valid_edges_count, out_features]
        h_dst = x_transformed[dst]  # Shape: [valid_edges_count, out_features]

        edge_features = torch.cat([h_src, h_dst, edge_attr_flat], dim=-1)  # Shape: [valid_edges_count, 2*out_features + edge_dim]
        
        e = torch.matmul(edge_features, self.a).squeeze(-1)  # Shape: [valid_edges_count]
        e = self.leaky_relu(e)
        


        softmax_values = torch.zeros_like(e)

        # **Apply Softmax per "graph" or "dst"**
        if self.softmax_per == "graph":
            # Softmax per graph
            # Sort edges based on batch IDs
            edge_batch_ids, sort_indices = torch.sort(edge_batch_ids)
            src = src[sort_indices]
            h_src = h_src[sort_indices]
            dst = dst[sort_indices]
            h_dst = h_dst[sort_indices]
            e = e[sort_indices]

            # Compute the number of edges per batch
            batch_counts = torch.bincount(edge_batch_ids, minlength=batch_size)  # Shape: [batch_size]

            # Get start and end indices for each batch
            batch_offsets = torch.cumsum(batch_counts, dim=0)  # Shape: [batch_size]
            batch_offsets = torch.cat([torch.tensor([0], device=x.device), batch_offsets[:-1]])  # Shift to get start indices

            split_attention_scores = torch.split(e, batch_counts.tolist())  # Split per batch
            softmax_values_split = [F.softmax(batch_scores/self.T, dim=0) for batch_scores in split_attention_scores]
            softmax_values = torch.cat(softmax_values_split, dim=0)

        elif self.softmax_per == "dst":
            # Softmax per destination node
            # Sort edges based on destination node IDs
            dst, sort_indices = torch.sort(dst)
            edge_batch_ids = edge_batch_ids[sort_indices]
            src = src[sort_indices]
            h_src = h_src[sort_indices]
            h_dst = h_dst[sort_indices]
            e = e[sort_indices]
            # Compute the number of edges per node
            node_counts = torch.bincount(dst, minlength=len(global_node_ids))

            # Get start and end indices for each node
            node_offsets = torch.cumsum(node_counts, dim=0)  # Shape: [valid_nodes_count]
            node_offsets = torch.cat([torch.tensor([0], device=x.device), node_offsets[:-1]])  # Shift to get start indices

            split_attention_scores = torch.split(e, node_counts.tolist())  # Split per node
            softmax_values_split = [F.softmax(node_scores/self.T, dim=0) for node_scores in split_attention_scores]
            softmax_values = torch.cat(softmax_values_split, dim=0)

        # **9. Aggregate neighbor information using attention coefficients**
        h_aggregated = torch.zeros(len(global_node_ids), x_transformed.shape[-1], device=x.device)  # Shape: [valid_nodes_count, out_features]
        h_aggregated.index_add_(dim=0, index=dst, source=softmax_values.unsqueeze(-1) * h_src)

        # **10. Restore output to original shape considering node_mask**
        maps = global_to_batch_node_map[global_node_ids]
        x = torch.zeros_like(x)
        x[maps[:,0], maps[:,1]] = h_aggregated  # Assign values to valid nodes

        return x  # Shape: [batch_size, seq_len, out_features]

    @staticmethod
    def create_aggregate_to_new_root_edge(x, node_mask):
        valid_nodes = torch.nonzero(~node_mask, as_tuple=True)
        edge_idx = torch.full((*node_mask.shape, 2), -1.0, dtype=torch.int64, device=node_mask.device)
        edge_idx[valid_nodes[0], valid_nodes[1], 0] = valid_nodes[1]+1 # [[1, -1], [2, -1], ...]
        edge_idx[valid_nodes[0], valid_nodes[1], 1] = 0 # [[1, 0], [2, 0], ...]

        x = torch.cat([torch.zeros_like(x[...,[0], :]), x], dim=-2)
        node_mask = torch.cat([torch.zeros_like(node_mask[:,[0]]), node_mask], dim=-1)
        edge_mask = torch.full_like(node_mask, True)[..., :-1]
        edge_mask[valid_nodes[0], valid_nodes[1]] = False

        return x, edge_idx, node_mask, edge_mask


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
    def attention(query, key, value, key_mask=None, dropout=None):
        """
        Calculate the scaled dot-product attention.
        
        Args:
            query (torch.Tensor): The query matrix Q.
            key (torch.Tensor): The key matrix K.
            value (torch.Tensor): The value matrix V.
            query_mask (torch.Tensor, optional): The mask to prevent attention to certain positions.
            dropout (nn.Dropout, optional): Dropout layer for attention scores.

        Returns:
            torch.Tensor: The attention-weighted output.
            torch.Tensor: The attention scores.
        """
        d_k = query.shape[-1]  # Dimension of each head
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if provided)
        if key_mask is not None:
            # Expand mask to match the attention scores dimensions
            # expand_key_mask = query_mask.unsqueeze(1).unsqueeze(2).transpose(2,3).expand(attention_scores.shape)
            expand_key_mask = key_mask.unsqueeze(1).unsqueeze(2).expand(attention_scores.shape)
            attention_scores = attention_scores.masked_fill(expand_key_mask, -1e9)

        # Apply softmax to normalize the scores
        attention_scores = torch.softmax(attention_scores, dim=-1)

        # if query_mask is not None:
        #     # Expand mask to match the attention scores dimensions
        #     expand_query_mask = query_mask.unsqueeze(3).unsqueeze(4).transpose(2,3).expand(attention_scores.shape)
        #     attention_scores = attention_scores.masked_fill(expand_query_mask == False, 0.0)

        # Apply dropout (if provided)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Compute the attention-weighted output
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, key_mask=None):
        """
        Forward pass of the Multi-Head Attention block.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor, optional): Mask tensor to apply on attention scores.

        Returns:
            torch.Tensor: Output tensor after multi-head attention.
        """
        # Linear projections for Q, K, V
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split the embeddings into h heads and reshape (batch_size, seq_len, embed_dim) --> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2).contiguous()
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2).contiguous()
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2).contiguous()

        # Compute attention
        x, self.attention_scores = self.attention(query, key, value, key_mask, self.dropout)

        # Concatenate all heads back together (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, embed_dim)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final linear transformation (batch_size, seq_len, embed_dim)
        return self.w_o(x)
    
    def get_attention_scores_mean(self):
        scores = self.attention_scores.transpose(1,2).contiguous() # (batch, query_seq_len, h, key_seq_len)
        mean = scores.mean(dim=2) # (batch, query_seq_len, key_seq_len)
        return mean

class GatEncoderBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, heads, ff_dim, dropout=0.1, T=1.0):
        super(GatEncoderBlock, self).__init__()
        self.gat = GATLayer(node_dim, node_dim, edge_dim, directed=True, softmax_per="dst", T=T)
        self.self_attention = MultiHeadAttentionBlock(node_dim, h=heads, dropout=dropout)
        # self.self_attention2 = MultiHeadAttentionBlock(node_dim, h=heads, dropout=dropout)
                
        # Feed Forward Block
        self.feed_forward = FeedForwardBlock(node_dim, ff_dim, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.norm3 = nn.LayerNorm(node_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, edge_index, edge_attr, node_mask, edge_mask):
        output = self.gat(h, edge_index, edge_attr, node_mask, edge_mask)
        h = self.norm1(self.dropout(output))  # Apply GAT block

        output = self.self_attention(h, x, x, key_mask=node_mask)
        h = self.norm2(h + self.dropout(output))  # Apply self-attention block

        output = self.feed_forward(h)
        h = self.norm3(self.dropout(output))

        return h  # Shape: [batch_size, seq_len, node_dim]

class GatDecoderBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, heads, ff_dim, dropout=0.1, T=1.0):
        super(GatDecoderBlock, self).__init__()
        self.gat = GATLayer(node_dim, node_dim, edge_dim, directed=True, softmax_per="dst", T=T)
        self.self_attention = MultiHeadAttentionBlock(node_dim, h=heads, dropout=dropout)
        self.cross_attention = MultiHeadAttentionBlock(node_dim, h=heads, dropout=dropout)
                
        # Feed Forward Block
        self.feed_forward = FeedForwardBlock(node_dim, ff_dim, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.norm3 = nn.LayerNorm(node_dim)
        self.norm4 = nn.LayerNorm(node_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, edge_index, edge_attr, enc_output, tgt_mask, memory_mask, edge_mask):
        # Shape of x [batch_size, seq_len, node_dim]
        # Shape of h [batch_size, seq_len, node_dim]
        # Shape of enc_output [batch_size, encoder_seq_len, dim]
        output = self.gat(h, edge_index, edge_attr, tgt_mask, edge_mask)
        h = self.norm1(self.dropout(output))  # Apply GAT block

        output = self.self_attention(h, x, x, key_mask=tgt_mask)
        h = self.norm2(h + self.dropout(output))  # Apply self-attention block

        output = self.cross_attention(h, enc_output, enc_output, key_mask=memory_mask)
        h = self.norm3(h + self.dropout(output)) # Apply encoder-decoder attention block

        output = self.feed_forward(h)
        h = self.norm4(self.dropout(output))

        return h  # Shape: [batch_size, seq_len, node_dim]

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttentionBlock(embed_dim, h=heads, dropout=dropout)
        self.cross_attention = MultiHeadAttentionBlock(embed_dim, h=heads, dropout=dropout)
                
        # Feed Forward Block
        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, enc_output, tgt_mask, memory_mask):
        # Shape of x [batch_size, seq_len, embed_dim]
        # Shape of h [batch_size, seq_len, embed_dim]
        # Shape of enc_output [batch_size, encoder_seq_len, embed_dim]
        output = self.self_attention(h, x, x, key_mask=tgt_mask)
        h = self.norm1(h + self.dropout(output))  # Apply self-attention block

        output = self.cross_attention(h, enc_output, enc_output, key_mask=memory_mask)
        h = self.norm2(h + self.dropout(output)) # Apply encoder-decoder attention block

        output = self.feed_forward(h)
        h = self.norm3(self.dropout(output))

        return h  # Shape: [batch_size, seq_len, embed_dim]

def get_flatten_layer(flatten_info, in_features, out_features):
    if flatten_info['mode'] == 'gat_aggregate':
        mode = 'gat_aggregate'
        flatten_layer = GatAggregateFlatten(
            in_features=in_features,
            out_features=out_features,
            T=flatten_info['gat_T'],
        )
        return mode, flatten_layer
    else:
        raise NotImplementedError("Flatten layer mode not implemented.")

class GatAggregateFlatten(nn.Module):
    def __init__(self, in_features, out_features, T=1.0):
        super(GatAggregateFlatten, self).__init__()
        directed = True
        softmax_per = "dst"
        self.aggregate_layer = GATLayer(in_features, out_features, edge_dim=0, directed=directed, softmax_per=softmax_per, T=T)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, node_mask):
        # shape of x [batch_size, seq_len, in_features]
        x, edge_index, node_mask, edge_mask = GATLayer.create_aggregate_to_new_root_edge(x, node_mask)
        edge_attr = torch.empty((*edge_index.shape[:-1], 0), dtype=torch.float32, device=x.device)
        x = self.aggregate_layer(x, edge_index, edge_attr, node_mask, edge_mask)
        return self.norm(x[..., 0, :])  # Shape: [batch_size, out_features]

class SelectFlatten(nn.Module):
    def __init__(self, embed_dim):
        super(SelectFlatten, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, select_indices):
        # shape of x [batch_size, seq_len, embed_dim]
        x = x[..., select_indices, :]
        return self.norm(x)  # Shape: [batch_size, embed_dim]

# Not implemented Flatten layers
if False:
    class Conv1dFlatten(nn.Module):
        def __init__(self, embed_dim, out_features, seq_len, conv_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
            super(Conv1dFlatten, self).__init__()
            
            layers = []
            in_channels = embed_dim
            for out_channels in conv_channels:
                layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)), 
                layers.append(nn.ReLU())
                in_channels = out_channels
            
            # 最終的な畳み込み層で必要な出力特徴量に合わせる
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_features, kernel_size=seq_len))

            self.conv_layers = nn.Sequential(*layers)

        def forward(self, x):
            x = x.transpose(1, 2)  # (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
            x = self.conv_layers(x) # (batch, out_features, 1)
            x = x.squeeze(2) # (batch_size, out_features, 1) -> (batch_size, out_features)
            return x
        


    class LinearFlatten(nn.Module):
        def __init__(self, embed_dim, out_features, hidden_dim:list[int]=[]):
            super(LinearFlatten, self).__init__()
            
            layers = []
            in_channels = embed_dim
            nodes = [embed_dim] + hidden_dim + [out_features]
            for i in range(len(nodes) - 2):
                layers.append(nn.Linear(in_features=nodes[i], out_features=nodes[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(in_features=nodes[-2], out_features=nodes[-1]))

            self.linear_layers = nn.Sequential(*layers)

        def forward(self, x):
            # Apply average pooling over the sequence length dimension
            x = x.mean(dim=1)  # Shape: [batch_size, embed_dim]

            # Linear transformation: [batch_size, embed_dim] -> [batch_size, out_features]
            x = self.linear_layers(x)
            return x

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
    
if __name__ == "__main__":
    # Parameters
    batch_size = 2
    seq_len = 5
    embed_dim = 4
    num_heads = 2

    # Sample input: [batch_size, seq_len, embed_dim]
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Key padding mask: [batch_size, seq_len]
    # True = padding position (ignore), False = valid position (attend to)
    key_padding_mask = torch.tensor([[False, False, True, True, True],  # Only first 2 tokens are valid
                                    [False, False, False, True, True]]) # Only first 3 tokens are valid

    # Define multi-head attention
    multi_head_attn = MultiHeadAttentionBlock(embed_dim=embed_dim, h=num_heads, dropout=0.1)

    # Apply multi-head attention with key_padding_mask
    attn_output, attn_weights = multi_head_attn(x, x, x, mask=key_padding_mask)

    # Print results
    print("Attention Output:")
    print(attn_output)

    print("\nAttention Weights:")
    print(attn_weights)