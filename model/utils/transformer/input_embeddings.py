import torch
import torch.nn as nn
from .graph_lib import LSTMGraph
from torch_geometric.data import Batch, Data


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
