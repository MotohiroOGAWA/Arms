import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from typing import NamedTuple
import inspect

import torch
import torch.nn as nn
from tqdm import tqdm

from .graph_lib import *
from .chem_encoder import HierGatChemEncoder
from .ms_encoder import MsCrossEncoder
from .decoder import HierGATDecoder
from .transformer import *

class Ms2z(nn.Module):
    def __init__(
            self, vocab_data, max_seq_len, atom_layer_lstm_iterations,
            chem_node_dim, chem_edge_dim, chem_hidden_dim, fp_subspace_dim, alpha_attachment_dim, 
            fp_dropout,
            precursor_hidden_dims:list[int], fragment_hidden_dims:list[int], ms_embed_dim, ms_hidden_dim, ms_embed_dropout:float,
            chem_encoder_use_shared_block, chem_encoder_layers, 
            chem_encoder_heads, chem_encoder_ff_dim, chem_encoder_dropout, chem_encoder_gat_T,
            chem_encoder_flatten,
            ms_encoder_use_shared_block, ms_encoder_layers,
            ms_encoder_heads, ms_encoder_ff_dim, ms_encoder_dropout,
            ms_encoder_precursor_seq_len,
            ms_encoder_flatten,
            latent_dim,
            memory_seq_len,
            decoder_use_shared_block, decoder_layers, 
            decoder_heads, decoder_ff_dim, decoder_dropout, decoder_gat_T,
            decoder_flatten,
            ):
        assert (chem_node_dim+chem_edge_dim) % decoder_heads == 0, "node_dim+edge_dim must be divisible by decoder_heads."
        super(Ms2z, self).__init__()
        attached_motif_index_map = vocab_data['attached_motif_index_map']
        bonding_cnt_tensor = vocab_data['bonding_cnt_tensor']
        atom_layer_list = vocab_data['atom_layer_list']
        motif_fp_tensor = vocab_data['motif_fp_tensor']
        attached_motif_fp_tensor = vocab_data['attached_motif_fp_tensor']
        connect_to_fp_tensor = vocab_data['connect_to_fp_tensor']
        bos_idx = vocab_data['bos']
        pad_idx = vocab_data['pad']
        unk_idx = vocab_data['unk']
        fragment_feature_size = len(vocab_data['feature_name_to_idx'])
        precurosor_feature_size = len(vocab_data['precursor_feature_name_to_idx'])

        self.is_sequential = False # Sequential processing

        constructor_params = inspect.signature(self.__init__).parameters
        config_keys = [param for param in constructor_params if param != "self"]
        for key in config_keys:
            setattr(self, key, locals()[key])

        # Vocabulary embedding
        self.frag_embedding = FragEmbeddings(
            chem_node_dim, chem_edge_dim,
            fp_subspace_dim,
            attached_motif_index_map, 
            bonding_cnt_tensor, 
            atom_layer_list,
            atom_layer_lstm_iterations,
            motif_fp_tensor,
            attached_motif_fp_tensor,
            connect_to_fp_tensor,
            fp_dropout,
            bos_idx, pad_idx, unk_idx,
            )
        
        self.motif_size = self.frag_embedding.motif_size
        self.att_size = self.frag_embedding.att_size
        self.max_bonding_cnt = self.frag_embedding.max_bonding_cnt

        # ms embedding
        self.ms_embedding = MsEmbeddings(
            precursor_dim=precurosor_feature_size,
            precursor_hidden_dims=precursor_hidden_dims,
            fragment_dim=fragment_feature_size,
            fragment_hidden_dims=fragment_hidden_dims,
            embed_dim=ms_embed_dim,
            dropout=ms_embed_dropout,
        )

        self.bos = nn.Parameter(torch.tensor(bos_idx, dtype=torch.int32), requires_grad=False)
        self.attached_bos_idx = nn.Parameter(torch.tensor([self.bos, 0], dtype=torch.int64), requires_grad=False)
        self.pad = nn.Parameter(torch.tensor(pad_idx, dtype=torch.int32), requires_grad=False)
        self.attached_pad_idx = nn.Parameter(torch.tensor([self.pad, 0], dtype=torch.int64), requires_grad=False)
        self.unk = nn.Parameter(torch.tensor(unk_idx, dtype=torch.int32), requires_grad=False)
        self.attached_unk_idx = nn.Parameter(torch.tensor([self.unk, 0], dtype=torch.int64), requires_grad=False)

        # Chem encoder
        self.chem_encoder = HierGatChemEncoder(
            num_layers=chem_encoder_layers,
            node_dim=chem_node_dim+chem_edge_dim,
            edge_dim=chem_edge_dim,
            hidden_dim=chem_hidden_dim,
            num_heads=chem_encoder_heads,
            ff_dim=chem_encoder_ff_dim,
            dropout=chem_encoder_dropout,
            T=chem_encoder_gat_T,
            use_shared_block=chem_encoder_use_shared_block,
        )

        # Chem Flatten layer
        self.chem_encoder_flatten_mode, self.chem_encoder_flatten_layer \
            = get_flatten_layer(
                flatten_info=chem_encoder_flatten,
                in_features=chem_hidden_dim,
                out_features=chem_hidden_dim,
            )

        # Chem Latent sampler
        self.chem_latent_sampler = LatentSampler(
            input_dim=chem_hidden_dim,
            latent_dim=latent_dim,
        )

        # MS encoder
        self.ms_encoder = MsCrossEncoder(
            num_layers=ms_encoder_layers,
            embed_dim=ms_embed_dim,
            hidden_dim=ms_hidden_dim,
            num_heads=ms_encoder_heads,
            ff_dim=ms_encoder_ff_dim,
            precursor_seq_len=ms_encoder_precursor_seq_len,
            dropout=ms_encoder_dropout,
            use_shared_block=ms_encoder_use_shared_block,
        )
        
        # MS Flatten layer
        self.ms_encoder_flatten_mode, self.ms_encoder_flatten_layer \
            = get_flatten_layer(
                flatten_info=ms_encoder_flatten,
                in_features=ms_hidden_dim,
                out_features=chem_hidden_dim,
            )

        # MS Latent sampler
        self.ms_latent_sampler = LatentSampler(
            input_dim=chem_hidden_dim,
            latent_dim=latent_dim,
        )

        self.memory_linear = nn.Linear(latent_dim, chem_hidden_dim*memory_seq_len)

        # Transformer decoder
        self.decoder = HierGATDecoder(
            num_layers=decoder_layers,
            node_dim=chem_node_dim+chem_edge_dim,
            edge_dim=chem_edge_dim,
            hidden_dim=chem_hidden_dim,
            num_heads=decoder_heads,
            ff_dim=decoder_ff_dim,
            dropout=decoder_dropout,
            T=decoder_gat_T,
            use_shared_block=decoder_use_shared_block,
        )
        
        # Decoder Flatten layer
        self.decoder_flatten_mode, self.decoder_flatten_layer \
            = get_flatten_layer(
                flatten_info=decoder_flatten,
                in_features=chem_hidden_dim,
                out_features=chem_hidden_dim,
            )
        

        # Loss
        self.motif_embed_dim = chem_node_dim
        self.attached_motif_embed_dim = chem_node_dim+alpha_attachment_dim
        self.predict_linear = nn.Sequential(
            nn.Linear(chem_hidden_dim, chem_hidden_dim),
            nn.ReLU(),
            nn.Linear(chem_hidden_dim, self.attached_motif_embed_dim+chem_edge_dim),
        )
        
        # self.pred_token_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='none')
        
        # predict motif
        self.pred_motif_layer = nn.Sequential(
            nn.Linear(chem_node_dim, chem_node_dim),
            nn.ReLU(),
            nn.Linear(chem_node_dim, self.motif_size),
        )
        self.pred_motif_loss_fn = F.cross_entropy

        # predict attachment
        self.pred_attachment_layer = nn.Sequential(
            nn.Linear(self.attached_motif_embed_dim, self.attached_motif_embed_dim),
            nn.ReLU(),
            nn.Linear(self.attached_motif_embed_dim, chem_node_dim),
        )
        self.pred_attachment_attention_layer \
            = MultiHeadAttentionBlock(
                embed_dim=chem_node_dim,
                h=decoder_heads,
                dropout=decoder_dropout,
            )
        self.pred_attachment_loss_fn = F.cross_entropy

        # predict root bond position
        self.pred_root_bond_pos_layer = nn.Sequential(
            nn.Linear(chem_edge_dim, chem_edge_dim),
            nn.ReLU(),
            nn.Linear(chem_edge_dim, chem_edge_dim),
        )
        self.pred_root_bond_pos_attention_layer \
            = MultiHeadAttentionBlock(
                embed_dim=chem_edge_dim,
                h=decoder_heads,
                dropout=decoder_dropout,
            )
        self.pred_root_bond_pos_loss_fn = F.cross_entropy

        # latent space to fingerprint loss
        fp_dim = self.frag_embedding.fp_dim
        self.fp_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.latent_to_fp_layer = nn.Sequential(
            nn.Linear(latent_dim, 2*fp_dim),
            nn.ReLU(),
            nn.Linear(2*fp_dim, fp_dim),
        )

    def forward_only_chem_encoder(self, input_tensor, target_tensor):
        """
        Forward pass for Ms2z model.

        Args:
            token_tensor (torch.Tensor): Token input tensor. [batch_size, seq_len, 2 (motif, attachment)]
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. [batch_size, seq_len, 3 (parent_idx, parent_bond_pos, bond_pos)]
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.
        """
        # assert 'token' in target_tensor and 'order' in target_tensor and 'mask' in target_tensor, "target tensor must contain 'token', 'order', and 'mask' keys."
        is_onnx = False
        if not isinstance(input_tensor, ModelChemInput) and isinstance(input_tensor, tuple):
            input_tensor = ModelChemInput.from_tuple(input_tensor)
            is_onnx = True
        if not isinstance(target_tensor, ModelChemInput) and isinstance(target_tensor, tuple):
            target_tensor = ModelChemInput.from_tuple(target_tensor)
            is_onnx = True

        # self.frag_embedding.reset_calc_embeddings()

        # Prepare input
        if self.use_chem_encoder:
            node_embed, edge_index, edge_attr_embed, mask_node, mask_edge \
                = self.prepare_chem_input(*[getattr(input_tensor, field) for field in ['token', 'order', 'mask']])
            encoder_output = self.chem_encoder(
                x=node_embed,
                edge_index=edge_index,
                edge_attr=edge_attr_embed,
                node_mask=mask_node,
                edge_mask=mask_edge,
            )
            z, mean, log_var = self.chem_latent_sampler(encoder_output)
            
        else:
            pass

        # Prepare decoder
        if self.is_sequential:
            pass
        else:
            node_embed_flat, edge_index_flat, edge_attr_embed_flat, \
                tgt_idx_flat, mask_seq_flat, mask_edge_flat, batch_idx_flat, unk_idx_flat = \
                self.prepare_batch_decoder(*[getattr(target_tensor, field) for field in ['token', 'order', 'mask']])
            
        # Transformer decoder
        memory = z # [batch_size, latent_dim]
        # memory = z # [batch_size, latent_dim]
        memory = self.memory_linear(memory) # [batch_size, (node_dim+edge_dim)*memory_seq_len]
        memory_flat = memory[batch_idx_flat] # [valid_total, (node_dim+edge_dim)*memory_seq_len]
        memory_flat = memory_flat.reshape(*memory_flat.shape[:-1], -1, self.memory_seq_len) # [valid_total, node_dim+edge_dim, memory_seq_len]
        memory_flat = memory_flat.transpose(1, 2) # [valid_total, memory_seq_len, node_dim+edge_dim]

        # Decode using Transformer Decoder
        memory_mask_flat = torch.zeros(*memory_flat.shape[:2], dtype=torch.bool, device=memory_flat.device) # [valid_total, memory_seq_len, node_dim+edge_dim]

        # latent to fingerprint loss
        z_fp = self.latent_to_fp_layer(mean) # [batch_size, latent_dim]
        target_fp = input_tensor.fp
        latent_fp_loss, latent_fp_acc, latent_strict_fp_acc = self.calc_fp_loss(z_fp, target_fp, strict_threshold=0.1)

        if not is_onnx:
            loss_list = {
                'latent_fp': latent_fp_loss,
            }
            acc_list = {
                'latent_fp': latent_fp_acc,
            }
            target_data = {
                'latent_fp': {'loss': latent_fp_loss.item(), 'accuracy': latent_fp_acc, 'criterion': get_criterion_name(self.fp_loss_fn)},
            }

            return loss_list, acc_list, target_data
        else:
            pass
            # return pred_motif_loss
        
    def forward(self, input_chem_tensor, target_chem_tensor, input_ms_tensor, target_ms_tensor):
        """
        Forward pass for Ms2z model.

        Args:
            token_tensor (torch.Tensor): Token input tensor. [batch_size, seq_len, 2 (motif, attachment)]
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. [batch_size, seq_len, 3 (parent_idx, parent_bond_pos, bond_pos)]
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.
        """
        # assert 'token' in target_tensor and 'order' in target_tensor and 'mask' in target_tensor, "target tensor must contain 'token', 'order', and 'mask' keys."     
        is_onnx = False
        if not isinstance(input_chem_tensor, ModelChemInput) and isinstance(input_chem_tensor, tuple):
            input_chem_tensor = ModelChemInput.from_tuple(input_chem_tensor)
            is_onnx = True
        if not isinstance(target_chem_tensor, ModelChemInput) and isinstance(target_chem_tensor, tuple):
            target_chem_tensor = ModelChemInput.from_tuple(target_chem_tensor)
            is_onnx = True

        if len(input_chem_tensor) == 0:
            raise ValueError("Input Chem tensor is empty.")
        if len(target_chem_tensor) == 0:
            raise ValueError("Target Chem tensor is empty.")
        if len(input_ms_tensor) == 0:
            raise ValueError("Input MS tensor is empty.")
        if len(target_ms_tensor) == 0:
            raise ValueError("Target MS tensor is empty.")
        
        if input_chem_tensor.mask.sum(dim=-1).min() == 0:
            raise ValueError("Input Chem tensor contains 0 mask.")
        if input_chem_tensor.mask.sum(dim=-1).max() == input_chem_tensor.mask.size(1):
            raise ValueError("Input Chem tensor contains all mask.")
        if target_chem_tensor.mask.sum(dim=-1).min() == 0:
            raise ValueError("Target Chem tensor contains 0 mask.")
        if target_chem_tensor.mask.sum(dim=-1).max() == target_chem_tensor.mask.size(1):
            raise ValueError("Target Chem tensor contains all mask.")
        # if input_ms_tensor.fragment_mask.sum(dim=-1).min() == 0:
        #     raise ValueError("Input MS tensor contains 0 mask.")
        # if input_ms_tensor.fragment_mask.sum(dim=-1).max() == input_ms_tensor.fragment_mask.size(1):
        #     raise ValueError("Input MS tensor contains all mask.")
        if target_ms_tensor.mask.sum(dim=-1).min() == 0:
            raise ValueError("Target MS tensor contains 0 mask.")
        if target_ms_tensor.mask.sum(dim=-1).max() == target_ms_tensor.mask.size(1):
            raise ValueError("Target MS tensor contains all mask.")

        # Prepare chem input
        chem_z, chem_mean, chem_log_var = self.calc_chem_z(input_chem_tensor)

        # Prepare ms input
        ms_z, ms_mean, ms_log_var = self.calc_ms_z(input_ms_tensor)

        # Prepare decoder
        tgt_tensor = ModelChemInput.cat(target_chem_tensor, target_ms_tensor)
        decoder_input, batch_idx_flat, tgt_idx_flat = self.prepare_batch_decoder(*tgt_tensor.to_tuple())
            
        # Transformer decoder
        chem_batch_id = torch.arange(chem_z.size(0), device=chem_z.device)
        valid_chem_id = torch.nonzero(torch.isin(batch_idx_flat, chem_batch_id), as_tuple=True)[0]
        ms_batch_id = torch.arange(ms_z.size(0), device=ms_z.device)+chem_z.size(0)
        valid_ms_id = torch.nonzero(torch.isin(batch_idx_flat, ms_batch_id), as_tuple=True)[0]
        
        # Concatenate chem and ms latent variables
        memory = torch.cat([chem_z, ms_z], dim=0) # [batch_size, latent_dim]
        memory_flat = memory[batch_idx_flat] # [valid_total, node_dim+edge_dim]

        decoder_output_flat = self.calc_decoder_output(memory_flat, decoder_input)

        # Predict Chem
        chem_pred_motif_loss, chem_pred_motif_acc, \
        chem_pred_attachment_loss, chem_pred_attachment_acc, \
        chem_pred_root_bond_loss, chem_pred_root_bond_acc, \
            = self.calc_predict_loss(decoder_output_flat[valid_chem_id], tgt_idx_flat[valid_chem_id])
        
        # Predict MS
        ms_pred_motif_loss, ms_pred_motif_acc, \
        ms_pred_attachment_loss, ms_pred_attachment_acc, \
        ms_pred_root_bond_loss, ms_pred_root_bond_acc, \
            = self.calc_predict_loss(decoder_output_flat[valid_ms_id], tgt_idx_flat[valid_ms_id])
        
        chem_kl_divergence_loss = self.chem_latent_sampler.calc_kl_divergence(chem_mean, chem_log_var)
        ms_kl_divergence_loss = self.ms_latent_sampler.calc_kl_divergence(ms_mean, ms_log_var)

        # motif, attached motif, connect loss
        attached_motif_idx_flat = self.frag_embedding.attached_motif_index_map[tgt_idx_flat[:,0],tgt_idx_flat[:,1]] # Shape: [valid_total]
        motif_res, attached_motif_res, connect_res = self.frag_embedding.calc_embed_fp_loss(decoder_output_flat, attached_motif_idx_flat)

        # vocabulary embedding fp loss
        unique_attached_motif_idx = torch.unique(attached_motif_idx_flat)
        unique_attached_motif_embed = self.frag_embedding.attached_motif_embedding_layer(unique_attached_motif_idx)
        vocab_motif_fp_res, vocab_attached_motif_fp_res, vocab_connect_fp_res = self.frag_embedding.calc_embed_fp_loss(unique_attached_motif_embed, unique_attached_motif_idx)


        # # latent to fingerprint loss
        # z_fp = self.latent_to_fp_layer(mean) # [batch_size, latent_dim]
        # target_fp = input_tensor.fp
        # latent_fp_loss, latent_fp_acc, latent_strict_fp_acc = self.calc_fp_loss(z_fp, target_fp, strict_threshold=0.1)

        if not is_onnx:
            loss_list, acc_list, target_data = \
                self.loss_res(
                    chem_kl_divergence_loss, ms_kl_divergence_loss,
                    chem_pred_motif_loss, chem_pred_motif_acc, 
                    chem_pred_attachment_loss, chem_pred_attachment_acc, 
                    chem_pred_root_bond_loss, chem_pred_root_bond_acc,
                    ms_pred_motif_loss, ms_pred_motif_acc,
                    ms_pred_attachment_loss, ms_pred_attachment_acc,
                    ms_pred_root_bond_loss, ms_pred_root_bond_acc,
                    motif_res, attached_motif_res, connect_res,
                    vocab_motif_fp_res, vocab_attached_motif_fp_res, vocab_connect_fp_res,
                )
            return loss_list, acc_list, target_data
        else:
            return pred_motif_loss
    
    def calc_chem_z(self, input_chem_tensor):
        node_embed, edge_index, edge_attr_embed, mask_node, mask_edge \
            = self.prepare_chem_input(*[getattr(input_chem_tensor, field) for field in ['token', 'order', 'mask']])
        chem_encoder_output = self.chem_encoder(
            x=node_embed,
            edge_index=edge_index,
            edge_attr=edge_attr_embed,
            node_mask=mask_node,
            edge_mask=mask_edge,
        )
        if self.chem_encoder_flatten_mode == 'gat_aggregate':
            flatten_input = {'x': chem_encoder_output, 'node_mask': mask_node}
        chem_encoder_output = self.chem_encoder_flatten_layer(**flatten_input)
        chem_z, chem_mean, chem_log_var = self.chem_latent_sampler(chem_encoder_output)
        return chem_z, chem_mean, chem_log_var
    
    def calc_ms_z(self, input_ms_tensor):
        precursor_embed, fragment_embed = self.prepare_ms_input(*[getattr(input_ms_tensor, field) for field in ['precursor', 'fragment']]) 
        fragment_mask = input_ms_tensor.fragment_mask
        ms_encoder_output = self.ms_encoder(
            fragment_x=fragment_embed,
            precursor_x=precursor_embed,
            fragment_mask=fragment_mask,
        )
        if self.ms_encoder_flatten_mode == 'gat_aggregate':
            flatten_input = {'x': ms_encoder_output, 'node_mask': fragment_mask}
        ms_encoder_output = self.ms_encoder_flatten_layer(**flatten_input)
        ms_z, ms_mean, ms_log_var = self.ms_latent_sampler(ms_encoder_output)
        return ms_z, ms_mean, ms_log_var

    def calc_decoder_output(self, memory, decoder_input):
        memory = self.memory_linear(memory) # [batch_size, (node_dim+edge_dim)*memory_seq_len]
        memory = memory.reshape(*memory.shape[:-1], self.memory_seq_len, -1) # [valid_total, memory_seq_len, node_dim+edge_dim]

        # Decode using Transformer Decoder
        memory_mask_flat = torch.zeros(*memory.shape[:2], dtype=torch.bool, device=memory.device) # [valid_total, memory_seq_len, node_dim+edge_dim]

        # Decoder

        decoder_output = self.decoder(
            x=decoder_input.node_embed, 
            edge_index=decoder_input.edge_index, 
            edge_attr=decoder_input.edge_attr, 
            enc_output=memory, 
            tgt_mask=decoder_input.tgt_mask, 
            memory_mask=memory_mask_flat, 
            edge_mask=decoder_input.edge_mask,
        ) # Shape: [valid_total, seq_len, node_dim+edge_dim]
        
        # if self.decoder.aggregate_mode == 'unk':
        #     output_flat = decoder_output[torch.arange(decoder_output.size(0)), unk_idx_flat] # Shape: [valid_total, node_dim+edge_dim]
        # elif self.decoder.aggregate_mode == 'mean':
        #     output_flat = decoder_output.mean(dim=1) # Shape: [valid_total, node_dim+edge_dim]
        # elif self.decoder.aggregate_mode == 'new_root':
        #     output_flat = decoder_output[:,0]
        
        if self.decoder_flatten_mode == 'gat_aggregate':
            flatten_input = {'x': decoder_output, 'node_mask': decoder_input.tgt_mask}
        output = self.decoder_flatten_layer(**flatten_input)
        output = self.predict_linear(output) # Shape: [valid_total, node_dim+edge_dim]

        return output

    def pretrain_frag_embed_layer(self, batch_size):
        yield from self.frag_embedding.train_atom_layer_all(batch_size)
    
    def enable_sequential(self):
        self.is_sequential = True
    
    def disable_sequential(self):
        self.is_sequential = False

    def get_parent_attached_motif_idx(self, token_tensor, order_tensor):
        mask = order_tensor[..., 0] < 0
        valid_parent_idx = order_tensor[..., 0].masked_fill(mask, 0)
        parent_attached_motif_idx = torch.gather(
            token_tensor, dim=-2,
            index=valid_parent_idx.unsqueeze(-1).expand(*token_tensor.shape[:-1], 2)
        ) # [batch_size, seq_len, 2]
        parent_attached_motif_idx[mask] = self.attached_pad_idx.expand_as(parent_attached_motif_idx[mask])
        return parent_attached_motif_idx
    
    def get_chem_encoder_input_idx(self, token_tensor, order_tensor, mask_tensor):
        batch_size, seq_len, _ = token_tensor.shape

         # **1. Create padding and unknown order tensors**
        pad_order = torch.full((order_tensor.size(2),), -1, dtype=order_tensor.dtype, device=order_tensor.device)
        # true_mask = torch.tensor(True, dtype=mask_tensor.dtype, device=mask_tensor.device)

        # **3. Apply <PAD> token where mask is True**
        token_tensor[mask_tensor] = self.attached_pad_idx # Shape: [batch_size, seq_len, 2]
        order_tensor[mask_tensor] = pad_order # Shape: [batchsize, seq_len, 3]
        
        # cat node_idx and root_bond_pos
        node_idx = torch.cat([token_tensor, order_tensor[...,[2]]], dim=-1) # Shape: [batch_size, seq_len, 3]

        # **10. Construct edge indices for parent-child relationships**
        parent_idx = order_tensor[..., [0]] # Shape: [batch_size, seq_len, 1]
        parent_idx_safe = parent_idx.clone()
        parent_idx_safe[parent_idx_safe < 0] = 0
        parent_token_tensor = torch.gather(token_tensor, dim=1, index=parent_idx_safe.expand(-1, -1, 2)) # Shape: [batch_size, seq_len, 2]
        parent_token_tensor[:,0] = self.attached_pad_idx # Set <PAD> token for root node
        parent_token_tensor[mask_tensor] = self.attached_pad_idx # Set <PAD> token where mask is True

        # **11. Construct edge index tensor**
        mask_node = mask_tensor.clone() # Shape: [batch_size, seq_len]
        mask_edge = mask_tensor[:,1:] # Shape: [batch_size, seq_len-1]
        # `dst` → Parent nodes
        dst = parent_idx[:,1:]  # Shape: [batch_size, seq_len-1, 1]
        # `src` → Child nodes (current token index)
        src = torch.arange(1, seq_len, device=order_tensor.device, dtype=order_tensor.dtype).unsqueeze(0).repeat(dst.size(0), 1)  # Shape: [batch_size, seq_len-1]
        src[mask_edge] = -1
        src = src.unsqueeze(-1)  # Shape: [batch_size, seq_len-1, 1]
        edge_index = torch.cat([src, dst], dim=-1)  # Shape: [batch_size, seq_len-1, 2]

        # **12. Construct edge attribute index tensor**
        # `edge_attr_idx_flat` → Parent nodes and bond position
        edge_attr_idx = torch.cat([parent_token_tensor, order_tensor[...,[1]]], dim=-1) # Shape: [batch_size, seq_len, 3]
        edge_attr_idx = edge_attr_idx[:,1:] # Shape: [batch_size, seq_len-1, 3]

        return node_idx, edge_index, edge_attr_idx, mask_node, mask_edge
    
    def prepare_chem_input(self, token_tensor, order_tensor, mask_tensor):
        node_idx, edge_index, edge_attr_idx, mask_node, mask_edge \
        = self.get_chem_encoder_input_idx(token_tensor, order_tensor, mask_tensor)

        node_embed = self.frag_embedding(node_idx) # [batch_size, seq_len, node_dim+edge_dim]
        edge_attr_embed = self.frag_embedding.embed_edge_attr(edge_attr_idx) # [batch_size, seq_len, edge_dim]

        return node_embed, edge_index, edge_attr_embed, mask_node, mask_edge

    def prepare_lstm_chem_input(self, token_tensor, order_tensor, mask_tensor):
        attached_motif_idx = token_tensor[...,:] # [batch_size, seq_len, 2]
        root_bond_pos = order_tensor[...,2] # [batch_size, seq_len]
        attached_motif_idx_with_bond_pos = torch.cat([attached_motif_idx, root_bond_pos.unsqueeze(-1)], dim=-1) # [batch_size, seq_len, 3]
        node_embed_with_root_bond = self.frag_embedding(attached_motif_idx_with_bond_pos) # [batch_size, seq_len, node_dim+edge_dim]

        parent_attached_motif_idx = self.get_parent_attached_motif_idx(token_tensor, order_tensor)
        parent_bond_pos = order_tensor[...,1] # [batch_size, seq_len]
        parent_attached_motif_idx_with_bond_pos = torch.cat([parent_attached_motif_idx, parent_bond_pos.unsqueeze(-1)], dim=-1) # [batch_size, seq_len, 3]
        parent_bond_pos_embed = self.frag_embedding.embed_edge_attr(parent_attached_motif_idx_with_bond_pos) # [batch_size, seq_len, max_bonding_cnt]

        return node_embed_with_root_bond, parent_bond_pos_embed

    def prepare_ms_input(self, precursor_tensor, fragment_tensor):
        precursor_embed = self.ms_embedding(precursor_tensor, is_precursor=True)
        fragment_embed = self.ms_embedding(fragment_tensor, is_precursor=False)
        return precursor_embed, fragment_embed

    def prepare_batch_decoder(self, token_tensor, order_tensor, mask_tensor):
        node_idx_flat, edge_index_flat, edge_attr_idx_flat, \
            tgt_idx_flat, mask_seq_flat, mask_edge_flat, batch_idx_flat, unk_idx_flat = \
            self.get_decoder_input_idx(token_tensor, order_tensor, mask_tensor)
        
        node_embed_flat = self.frag_embedding(node_idx_flat) # [valid_total, seq_len, node_dim+edge_dim]
        edge_attr_embed_flat = self.frag_embedding.embed_edge_attr(edge_attr_idx_flat) # [valid_total, seq_len, edge_dim]

        decoder_input = ModelDecoderInput(
            node_embed=node_embed_flat,
            edge_index=edge_index_flat,
            edge_attr=edge_attr_embed_flat,
            tgt_mask=mask_seq_flat,
            edge_mask=mask_edge_flat,
        )

        return decoder_input, batch_idx_flat, tgt_idx_flat
    
    def prepare_sequential_decoder(self, token_tensor, order_tensor, mask_tensor):
        node_idx_flat, edge_index_flat, edge_attr_idx_flat, \
            mask_seq_flat, mask_edge_flat, unk_idx_flat = \
            self.get_decoder_input_idx(token_tensor, order_tensor, mask_tensor, is_sequential=True)
        
        node_embed_flat = self.frag_embedding(node_idx_flat) # [valid_total, seq_len, node_dim+edge_dim]
        edge_attr_embed_flat = self.frag_embedding.embed_edge_attr(edge_attr_idx_flat) # [valid_total, seq_len, edge_dim]

        decoder_input = ModelDecoderInput(
            node_embed=node_embed_flat,
            edge_index=edge_index_flat,
            edge_attr=edge_attr_embed_flat,
            tgt_mask=mask_seq_flat,
            edge_mask=mask_edge_flat,
        )

        return decoder_input, edge_index_flat, edge_attr_idx_flat, unk_idx_flat


    def get_decoder_input_idx(self, token_tensor, order_tensor, mask_tensor, is_sequential=False):
        batch_size, seq_len, _ = token_tensor.shape

         # **1. Create padding and unknown order tensors**
        pad_order = torch.full((order_tensor.size(2),), -1, dtype=order_tensor.dtype, device=order_tensor.device)
        unk_order = torch.full((order_tensor.size(2),), -1, dtype=order_tensor.dtype, device=order_tensor.device)
        false_mask = torch.tensor(False, dtype=mask_tensor.dtype, device=mask_tensor.device)
        # true_mask = torch.tensor(True, dtype=mask_tensor.dtype, device=mask_tensor.device)
        
        # **2. Add <BOS> token at the beginning of each sequence**
        token_seq_tensor = torch.cat([
            self.attached_bos_idx.repeat(batch_size, 1).unsqueeze(1),
            token_tensor
        ], dim=1)  # Shape: [batchsize, seq_len+1, 2]
        order_seq_tensor = torch.cat([
            pad_order.unsqueeze(0).repeat(batch_size, 1, 1),
            order_tensor
        ], dim=1)  # Shape: [batchsize, seq_len+1, 3]
        mask_seq_tensor = torch.cat([
            false_mask.unsqueeze(0).repeat(batch_size, 1),
            mask_tensor,
        ], dim=1)  # Shape: [batchsize, seq_len+1]

        # **3. Apply <PAD> token where mask is True**
        token_seq_tensor[mask_seq_tensor] = self.attached_pad_idx # Shape: [batchsize, seq_len+1, 2]
        order_seq_tensor[mask_seq_tensor] = pad_order # Shape: [batchsize, seq_len+1, 3]
        
        # **4. Create valid input mask for sequence expansion**
        if is_sequential:
            mask_tensor = mask_seq_tensor[:,:-1]

        valid_input_idx = ~create_expand_mask(mask_tensor, fill=True) # Shape: [batch_size, seq_len, seq_len]
        valid_input_idx = torch.cat([valid_input_idx, false_mask.expand(*valid_input_idx.shape[:2], 1)], dim=-1)  # Shape: [batch_size, seq_len, seq_len+1]
        
        # Expand valid indices for different dimensions
        valid_input_idx2 = valid_input_idx.unsqueeze(3).expand(-1, -1, -1, 2)  # Shape: [batch_size, seq_len, seq_len, 2]
        valid_input_idx3 = valid_input_idx.unsqueeze(3).expand(-1, -1, -1, 3)  # Shape: [batch_size, seq_len, seq_len, 3]

        # **5. Create unknown token mask for sequence expansion**
        valid_diagonal_idx = ~create_expand_mask(mask_tensor, fill=False)  # Shape: [batch_size, seq_len, seq_len]
        valid_unk_idx = torch.cat([false_mask.unsqueeze(0).unsqueeze(0).expand(batch_size,seq_len,1), valid_diagonal_idx], dim=-1)  # Shape: [batch_size, seq_len, seq_len+1]
        valid_unk_idx2 = valid_unk_idx.unsqueeze(3).expand(-1, -1, -1, 2)  # Shape: [batch_size, seq_len, seq_len+1, 2]
        valid_unk_idx3 = valid_unk_idx.unsqueeze(3).expand(-1, -1, -1, 3)  # Shape: [batch_size, seq_len, seq_len+1, 3]
        
        # Get valid sequence indices
        if is_sequential:
            no_mask_tensor = torch.zeros_like(mask_tensor)
            no_mask_tensor[torch.arange(batch_size, device=mask_tensor.device), (~mask_tensor).sum(dim=1)-1] = True
            valid_indices = no_mask_tensor.nonzero(as_tuple=True)  # (2 [batch, seq], valid_idx_total)
        else:
            valid_indices = (~mask_tensor).nonzero(as_tuple=True)  # (2 [batch, seq], valid_idx_total)
        # valid_indices = (~mask_tensor).nonzero(as_tuple=True)  # (2 [batch, seq], valid_idx_total)

        unk_idx_flat = torch.nonzero(valid_unk_idx[valid_indices[0], valid_indices[1]], as_tuple=True)[1]

        # **6. Compute mask and valid sequence lengths**
        mask_seq_flat = valid_input_idx[valid_indices[0], valid_indices[1]] # Shape: [valid_idx_total, seq_len+1]
        mask_seq_flat[:,-1] = True
        mask_seq_flat = ~mask_seq_flat.roll(1) # Shape: [valid_idx_total, seq_len+1]
        valid_length_flat = (~mask_seq_flat).sum(dim=-1) # Shape: [valid_idx_total]

        # **7. Expand token sequence with valid indices**
        token_seq_ex = self.attached_pad_idx.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, seq_len+1, 1)  # Shape: [batch_size, seq_len, seq_len+1, 2]
        token_seq_ex = torch.where(valid_input_idx2, token_seq_tensor.unsqueeze(1), token_seq_ex)  # Shape: [batch_size, seq_len, seq_len+1, 2]
        token_seq_ex = torch.where(valid_unk_idx2, self.attached_unk_idx.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, seq_len+1, 1), token_seq_ex)  # Shape: [batch_size, seq_len, seq_len+1, 2]
        token_seq_flat = token_seq_ex[valid_indices[0], valid_indices[1]] # Shape: [valid_idx_total, seq_len+1, 2]
        
        # **8. Expand order sequence with valid indices**
        order_seq_ex = pad_order.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, seq_len+1, 1)  # Shape: [batch_size, seq_len, seq_len+1, 3]
        order_seq_ex = torch.where(valid_input_idx3, order_seq_tensor.unsqueeze(1), order_seq_ex)  # Shape: [batch_size, seq_len, seq_len+1, 3]
        order_seq_ex = torch.where(valid_unk_idx3, unk_order.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, seq_len+1, 1), order_seq_ex)  # Shape: [batch_size, seq_len, seq_len+1, 3]
        order_seq_flat = order_seq_ex[valid_indices[0], valid_indices[1]+1] # Shape: [valid_idx_total, seq_len+1, 1]

        # cat node_idx and root_bond_pos
        node_idx_flat = torch.cat([token_seq_flat, order_seq_flat[...,[2]]], dim=-1) # Shape: [valid_idx_total, seq_len+1, 3]

        # **10. Construct edge indices for parent-child relationships**
        parent_idx_flat = order_seq_flat[..., [0]]+1 # Shape: [valid_idx_total, seq_len+1, 1]
        parent_token_idx_flat = torch.gather(token_seq_flat, dim=1, index=parent_idx_flat.expand(-1, -1, 2)) # Shape: [valid_idx_total, seq_len+1, 2]
        parent_idx_flat[:,0] = -1
        parent_idx_flat[mask_seq_flat] = -1
        parent_token_idx_flat[:,0] = self.attached_pad_idx
        parent_token_idx_flat[mask_seq_flat] = self.attached_pad_idx

        # **11. Construct edge index tensor**
        mask_edge_flat = mask_seq_flat[:,1:] # Shape: [valid_idx_total, seq_len]
        # `src` → Parent nodes
        src = parent_idx_flat[:,1:]  # Shape: [valid_total, seq_len, 1]
        # `dst` → Child nodes (current token index)
        dst = torch.arange(1, seq_len+1, device=order_tensor.device, dtype=order_tensor.dtype).unsqueeze(0).repeat(src.size(0), 1)  # Shape: [valid_total, seq_len]
        dst[mask_edge_flat] = -1
        dst = dst.unsqueeze(-1)  # Shape: [valid_total, seq_len, 1]
        edge_index_flat = torch.cat([src, dst], dim=-1)  # Shape: [valid_total, seq_len, 2]

        # **12. Construct edge attribute index tensor**
        # `edge_attr_idx_flat` → Parent nodes and bond position
        edge_attr_idx_flat = torch.cat([parent_token_idx_flat, order_seq_flat[...,[2]]], dim=-1) # Shape: [valid_idx_total, seq_len+1, 3]
        edge_attr_idx_flat = edge_attr_idx_flat[:,1:] # Shape: [valid_idx_total, seq_len, 3]

        # **13. Target indices for decoder prediction**
        tgt_attached_motif_idx_flat = token_seq_tensor[valid_indices[0], valid_length_flat-1] # Shape: [valid_idx_total, 2]
        tgt_root_bond_pos_flat = order_seq_tensor[valid_indices[0], valid_length_flat-1, 2].unsqueeze(-1) # Shape: [valid_idx_total,1]
        tgt_idx_flat = torch.cat([tgt_attached_motif_idx_flat, tgt_root_bond_pos_flat], dim=-1) # Shape: [valid_idx_total, 3]

        # **14. Get batch indices**
        batch_idx_flat = valid_indices[0] # Shape: [valid_idx_total]


        node_idx_flat = node_idx_flat[:, :-1, :] # Shape: [valid_idx_total, seq_len, 3]
        edge_index_flat = edge_index_flat[:, :-1, :] # Shape: [valid_idx_total, seq_len-1, 2]
        edge_attr_idx_flat = edge_attr_idx_flat[:, :-1, :] # Shape: [valid_idx_total, seq_len-1, 3]
        mask_seq_flat = mask_seq_flat[:, :-1] # Shape: [valid_idx_total, seq_len]
        mask_edge_flat = mask_edge_flat[:, :-1] # Shape: [valid_idx_total, seq_len-1]
        
        if is_sequential:
            return node_idx_flat, edge_index_flat, edge_attr_idx_flat, mask_seq_flat, mask_edge_flat, unk_idx_flat
        else:
            return node_idx_flat, edge_index_flat, edge_attr_idx_flat, tgt_idx_flat, mask_seq_flat, mask_edge_flat, batch_idx_flat, unk_idx_flat
    
    def predict_tree(self, z_tensor, batch_size, seq_len):
        device = z_tensor.device

        # **出力用 completed tensor**
        completed_token_tensor = torch.full((z_tensor.size(0), seq_len, 2), -1, device=device)
        completed_order_tensor = torch.full((z_tensor.size(0), seq_len, 3), -1, device=device)
        completed_mask_tensor = torch.full((z_tensor.size(0), seq_len), True, device=device)

        # **初期バッチのセットアップ**
        z_idx = 0 # 読み込みインデックス
        active_batch_size = min(batch_size, z_tensor.size(0))  # 初期バッチサイズ
        current_z_indices = torch.arange(active_batch_size, device=device)  # 現在のバッチのインデックス
        token_tensor, order_tensor, mask_tensor = self.sequentail_initial_tensor(active_batch_size, seq_len)
        unconnected_bond = torch.zeros(active_batch_size, seq_len, self.frag_embedding.max_bonding_cnt, 
                                    dtype=torch.bool, device=self.pad.device)

        pbar = tqdm(total=z_tensor.size(0), desc="Processing Trees", unit="sample")

        # **処理ループ (すべてのデータが完了するまで実行)**
        while z_idx < z_tensor.size(0):
            batch_z_tensor = z_tensor[current_z_indices].to(self.pad.device)
            token_tensor, order_tensor, mask_tensor, unconnected_bond = self.next_sequence(
                batch_z_tensor, token_tensor, order_tensor, mask_tensor, unconnected_bond
            )
            full_seq_len_mask = (mask_tensor.sum(dim=1) < 2)

            # **完了したバッチを確認**
            completed_batch_idx = unconnected_bond.sum(dim=-1).sum(dim=-1) == 0  # True のものが完了
            completed_batch_idx = completed_batch_idx | full_seq_len_mask
            num_completed = completed_batch_idx.sum().item()  # 完了したサンプル数

            if num_completed > 0:
                # **完了データを保存**
                completed_batch_idx = completed_batch_idx.to(device)
                completed_token_tensor[current_z_indices[completed_batch_idx]] = token_tensor[completed_batch_idx].to(device)
                completed_order_tensor[current_z_indices[completed_batch_idx]] = order_tensor[completed_batch_idx].to(device)
                completed_mask_tensor[current_z_indices[completed_batch_idx]] = mask_tensor[completed_batch_idx].to(device)

                # **完了データを削除**
                token_tensor = token_tensor[~completed_batch_idx]
                order_tensor = order_tensor[~completed_batch_idx]
                mask_tensor = mask_tensor[~completed_batch_idx]
                unconnected_bond = unconnected_bond[~completed_batch_idx]
                current_z_indices = current_z_indices[~completed_batch_idx.to(device)]

                active_batch_size -= num_completed  # バッチサイズを更新
                pbar.update(num_completed)  # プログレスバーを更新

                # **即座に新しいデータを補充**
                new_batch_count = min(num_completed, z_tensor.size(0) - (z_idx + active_batch_size))
                if new_batch_count > 0:
                    new_token_tensor, new_order_tensor, new_mask_tensor = self.sequentail_initial_tensor(new_batch_count, seq_len)
                    new_unconnected_bond = torch.zeros(new_batch_count, seq_len, self.frag_embedding.max_bonding_cnt, 
                                                    dtype=torch.bool, device=self.pad.device)

                    token_tensor = torch.cat([token_tensor, new_token_tensor], dim=0)
                    order_tensor = torch.cat([order_tensor, new_order_tensor], dim=0)
                    mask_tensor = torch.cat([mask_tensor, new_mask_tensor], dim=0)
                    unconnected_bond = torch.cat([unconnected_bond, new_unconnected_bond], dim=0)
                    current_z_indices = torch.cat([current_z_indices, torch.arange(new_batch_count, device=device)+z_idx+active_batch_size], dim=0)

                    active_batch_size += new_batch_count  # バッチサイズを元に戻す

                # **次のバッチへ進む**
                z_idx += num_completed  # 置き換えた分だけインデックスを進める

                # pbar.set_postfix(active_batch=active_batch_size, completed_samples=z_idx)

        pbar.close()
        return completed_token_tensor, completed_order_tensor, completed_mask_tensor



    def next_sequence(self, z_tensor, token_tensor, order_tensor, mask_tensor, unconnected_bond):
        decoder_input, edge_index, edge_attr_idx, unk_idx = self.prepare_sequential_decoder(token_tensor, order_tensor, mask_tensor)
        
        decoder_output = self.calc_decoder_output(z_tensor, decoder_input)

        motif_embed, att_motif_embed, root_bond_embed = self.split_decoder_output(decoder_output)

        predicted_motif_idx = self.predict_motif(motif_embed)
        predicted_attachment_idx = self.predict_attachment(att_motif_embed, predicted_motif_idx)
        predicted_root_bond_idx = self.predict_root_bonding(root_bond_embed, predicted_motif_idx, predicted_attachment_idx)
        predicted_root_bond_idx[unk_idx < 2] = -1

        batch_idx = torch.arange(z_tensor.size(0), device=z_tensor.device)
        token_tensor[batch_idx, unk_idx-1] = torch.cat([predicted_motif_idx.unsqueeze(-1), predicted_attachment_idx.unsqueeze(-1)], dim=-1)
        order_row = torch.cat([
            order_tensor[batch_idx, unk_idx-1][:,0].unsqueeze(-1), 
            order_tensor[batch_idx, unk_idx-1][:,1].unsqueeze(-1), 
            predicted_root_bond_idx.unsqueeze(-1),
            ], dim=-1)
        order_tensor[batch_idx, unk_idx-1] = order_row
        order_tensor[:,0,:] = -1
        mask_tensor[batch_idx, unk_idx-1] = False

        attached_motif_idx = self.frag_embedding.attached_motif_index_map[predicted_motif_idx, predicted_attachment_idx] # [batch_size]
        bond_cnt_tensor = self.frag_embedding.bonding_cnt_tensor[attached_motif_idx] # [batch_size]
        unconnected_bond[batch_idx, unk_idx-1] \
            = torch.arange(self.frag_embedding.max_bonding_cnt, device=mask_tensor.device).expand(batch_idx.size(0), -1) \
                < bond_cnt_tensor.unsqueeze(-1)
        unconnected_bond[predicted_root_bond_idx > -1, (unk_idx-1)[predicted_root_bond_idx > -1], predicted_root_bond_idx[predicted_root_bond_idx > -1]] = False

        valid = (order_tensor[batch_idx,unk_idx-1][:,0]>-1)
        unconnected_bond[batch_idx[valid], order_tensor[batch_idx,unk_idx-1][:,0][valid], order_tensor[batch_idx,unk_idx-1][:,1][valid]] = False

        # **1. 各バッチごとに `True` が最初に出る `seq` を取得**
        seq_idx = (unconnected_bond.any(dim=-1)).int().argmax(dim=1)  # Shape: (batch_size,)

        # **2. `seq_idx` に対応する `bond_idx` を取得**
        bond_idx = unconnected_bond[torch.arange(z_tensor.size(0)), seq_idx].int().argmax(dim=1)  # Shape: (batch_size,)

        true_mask = unconnected_bond.any(dim=(1, 2))  # `True` が存在するものだけ

        # order_tensor[true_mask, unk_idx[true_mask]][:, 0] = seq_idx[true_mask]
        # order_tensor[true_mask, unk_idx[true_mask]][:, 1] = bond_idx[true_mask]
        order_tensor.index_put_((true_mask, unk_idx[true_mask], torch.tensor([0], device=order_tensor.device)), seq_idx[true_mask])
        order_tensor.index_put_((true_mask, unk_idx[true_mask], torch.tensor([1], device=order_tensor.device)), bond_idx[true_mask])



        return token_tensor, order_tensor, mask_tensor, unconnected_bond

    def sequentail_initial_tensor(self, batch_size, seq_len):
        token_tensor = self.attached_pad_idx.repeat(batch_size, seq_len, 1)
        order_tensor = torch.full((batch_size, seq_len, 3), -1, dtype=torch.long, device=self.attached_pad_idx.device)
        mask_tensor = torch.full((batch_size, seq_len), True, dtype=torch.bool, device=self.attached_pad_idx.device)
        return token_tensor, order_tensor, mask_tensor
    
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

    def calc_predict_loss(self, decoder_output, tgt_idx):
        motif_embed, att_motif_embed, root_bond_embed = self.split_decoder_output(decoder_output)
        tgt_motif_idx = tgt_idx[:, 0] # Shape: [valid_total]
        tgt_motif_and_att_idx = tgt_idx[:, 0:2] # Shape: [valid_total, 2]
        
        # Calculate motif prediction loss
        pred_motif_loss, pred_motif_acc = self.calc_predict_motif_loss(motif_embed, tgt_motif_idx)

        # Calculate attachment prediction loss
        pred_attachment_loss, pred_attachment_acc = self.calc_predict_attachment_loss(att_motif_embed, tgt_motif_and_att_idx)

        # Calculate root bonding position prediction loss
        pred_root_bond_loss, pred_root_bond_acc = self.calc_predict_root_bonding_loss(root_bond_embed, tgt_idx)

        return pred_motif_loss, pred_motif_acc, pred_attachment_loss, pred_attachment_acc, pred_root_bond_loss, pred_root_bond_acc

    def calc_predict_motif_loss(self, node_embed, tgt_motif_id):
        # Project node embeddings to vocabulary logits
        logits_vocab = self.pred_motif_layer(node_embed)  # Shape: [data_size, node_dim] -> [data_size, vocab_size]
        # Apply Softmax to get probabilities
        # probabilities = F.softmax(logits_vocab, dim=-1)  # Shape: [data_size, vocab_size]
        
        losses = self.pred_motif_loss_fn(logits_vocab, tgt_motif_id, reduction='none')

        # Compute weights for each token
        unique_tokens, counts = torch.unique(tgt_motif_id, return_counts=True)
        token_weights = (1 / counts.float()) / unique_tokens.size(0)

        # Create a weight tensor for each target
        weight_map = {token.item(): weight for token, weight in zip(unique_tokens, token_weights)}
        weights = torch.tensor([weight_map[token.item()] for token in tgt_motif_id], device=node_embed.device)  # Shape: [data_size]

        # Apply weights to the individual losses
        weighted_losses = losses * weights  # Shape: [data_size]

        # Compute the weighted average loss
        loss = weighted_losses.sum()

        # Compute the accuracy
        predicted_token_id = torch.argmax(logits_vocab, dim=1)
        correct = torch.eq(predicted_token_id, tgt_motif_id).to(dtype=torch.float32)

        # accuracy = correct.mean()
        accuracy = (correct * weights).sum()

        return loss, accuracy

    def predict_motif(self, node_embed):
        # Project node embeddings to vocabulary logits
        logits_vocab = self.pred_motif_layer(node_embed)  # Shape: [data_size, node_dim] -> [data_size, vocab_size]
        # Apply Softmax to get probabilities
        # probabilities = F.softmax(logits_vocab, dim=-1)  # Shape: [data_size, vocab_size]

        # Compute the accuracy
        predicted_motif_id = torch.argmax(logits_vocab, dim=1)

        return predicted_motif_id

    def calc_predict_attachment_loss(self, node_embed, motif_attachment_idx):
        motif_idx = motif_attachment_idx[...,0]
        attachment_idx = motif_attachment_idx[...,1]
        candidate_att_idx = self.frag_embedding.attached_motif_index_map[motif_idx] # [batch_size, max_bonding_cnt]
        candidate_att_mask = candidate_att_idx < 0
        candidate_att_embed = self.frag_embedding.attached_motif_embedding_layer(candidate_att_idx.masked_fill(candidate_att_mask, self.pad)) # [batch_size, max_bonding_cnt, node_dim]
        
        # Project node embeddings to vocabulary logits
        x = self.pred_attachment_layer(node_embed)  # Shape: [data_size, node_dim] -> [data_size, node_dim]
        x = self.pred_attachment_attention_layer(x.unsqueeze(1), candidate_att_embed, candidate_att_embed, key_mask=candidate_att_mask)  # Shape: [data_size, node_dim] -> [data_size, node_dim]
        attention_scores = self.pred_attachment_attention_layer.get_attention_scores_mean() # [batch_size, 1, max_bonding_cnt]
        attention_scores = attention_scores.squeeze(1) # [batch_size, max_bonding_cnt]

        # Compute the loss
        loss = self.pred_attachment_loss_fn(attention_scores, attachment_idx)

        # Compute the accuracy
        predicted_attachment_idx = torch.argmax(attention_scores, dim=1)
        correct = torch.eq(predicted_attachment_idx, attachment_idx).to(dtype=torch.float32)
        accuracy = correct.mean()
        
        return loss, accuracy
    
    def predict_attachment(self, node_embed, motif_idx):
        candidate_att_idx = self.frag_embedding.attached_motif_index_map[motif_idx] # [batch_size, max_bonding_cnt]
        candidate_att_mask = candidate_att_idx < 0
        candidate_att_embed = self.frag_embedding.attached_motif_embedding_layer(candidate_att_idx.masked_fill(candidate_att_mask, self.pad)) # [batch_size, max_bonding_cnt, node_dim]
        
        # Project node embeddings to vocabulary logits
        x = self.pred_attachment_layer(node_embed)  # Shape: [data_size, node_dim] -> [data_size, node_dim]
        x = self.pred_attachment_attention_layer(x.unsqueeze(1), candidate_att_embed, candidate_att_embed, key_mask=candidate_att_mask)  # Shape: [data_size, node_dim] -> [data_size, node_dim]
        attention_scores = self.pred_attachment_attention_layer.get_attention_scores_mean() # [batch_size, 1, max_bonding_cnt]
        attention_scores = attention_scores.squeeze(1) # [batch_size, max_bonding_cnt]

        # Compute the accuracy
        predicted_attachment_idx = torch.argmax(attention_scores, dim=1)
        
        return predicted_attachment_idx

    def calc_predict_root_bonding_loss(self, edge_embed, tgt_motif_attachment_bond_pos):
        motif_idx = tgt_motif_attachment_bond_pos[...,0]
        attachment_idx = tgt_motif_attachment_bond_pos[...,1]
        bond_pos = tgt_motif_attachment_bond_pos[...,2]

        attached_motif_idx = self.frag_embedding.attached_motif_index_map[motif_idx, attachment_idx] # [batch_size]
        # attached_motif_embed = self.frag_embedding.attached_motif_embedding_layer(attached_motif_idx) # [batch_size, node_dim]
        connect_tensor, valid_mask = self.frag_embedding.get_connet_tensor(attached_motif_idx) # [batch_size, max_bonding_cnt]
        connect_embed = self.frag_embedding.connect_linear(connect_tensor) # [batch_size, max_bonding_cnt, edge_dim]

        # filter root node
        mask_root_node = bond_pos < 0
        bond_pos = bond_pos[~mask_root_node]
        # attached_motif_embed = attached_motif_embed[~mask_root_node]
        connect_embed = connect_embed[~mask_root_node]
        valid_mask = valid_mask[~mask_root_node]
        edge_embed = edge_embed[~mask_root_node]

        # Project edge embeddings to vocabulary logits
        x = self.pred_root_bond_pos_layer(edge_embed)  # Shape: [data_size, edge_dim] -> [data_size, edge_dim]
        x = self.pred_root_bond_pos_attention_layer(x.unsqueeze(1), connect_embed, connect_embed, key_mask=~valid_mask)  # Shape: [data_size, edge_dim] -> [data_size, edge_dim]
        attention_scores = self.pred_root_bond_pos_attention_layer.get_attention_scores_mean() # [batch_size, 1, max_bonding_cnt]
        attention_scores = attention_scores.squeeze(1) # [batch_size, max_bonding_cnt]

        # Compute the loss
        loss = self.pred_root_bond_pos_loss_fn(attention_scores, bond_pos)

        # Compute the accuracy
        predicted_bond_pos = torch.argmax(attention_scores, dim=1)
        correct = torch.eq(predicted_bond_pos, bond_pos).to(dtype=torch.float32)
        accuracy = correct.mean()
        
        return loss, accuracy
    
    def predict_root_bonding(self, edge_embed, motif_idx, attachment_idx):

        attached_motif_idx = self.frag_embedding.attached_motif_index_map[motif_idx, attachment_idx] # [batch_size]
        # attached_motif_embed = self.frag_embedding.attached_motif_embedding_layer(attached_motif_idx) # [batch_size, node_dim]
        connect_tensor, valid_mask = self.frag_embedding.get_connet_tensor(attached_motif_idx) # [batch_size, max_bonding_cnt]
        connect_embed = self.frag_embedding.connect_linear(connect_tensor) # [batch_size, max_bonding_cnt, edge_dim]

        # Project edge embeddings to vocabulary logits
        x = self.pred_root_bond_pos_layer(edge_embed)  # Shape: [data_size, edge_dim] -> [data_size, edge_dim]
        x = self.pred_root_bond_pos_attention_layer(x.unsqueeze(1), connect_embed, connect_embed, key_mask=~valid_mask)  # Shape: [data_size, edge_dim] -> [data_size, edge_dim]
        attention_scores = self.pred_root_bond_pos_attention_layer.get_attention_scores_mean() # [batch_size, 1, max_bonding_cnt]
        attention_scores = attention_scores.squeeze(1) # [batch_size, max_bonding_cnt]

        predicted_root_bond_pos = torch.argmax(attention_scores, dim=1)
        
        return predicted_root_bond_pos

    def split_decoder_output(self, decoder_output):
        motif_embed = decoder_output[:, :self.motif_embed_dim] # Shape: [valid_total, node_dim]
        att_motif_embed = decoder_output[:, :self.attached_motif_embed_dim] # Shape: [valid_total, node_dim+alpha_attachment_dim]
        root_bond_embed = decoder_output[:, self.attached_motif_embed_dim:] # Shape: [valid_total, edge_dim]
        return motif_embed, att_motif_embed, root_bond_embed

    def loss_res(self,
                        chem_kl_divergence_loss, ms_kl_divergence_loss,
                        chem_pred_motif_loss, chem_pred_motif_acc, 
                        chem_pred_attachment_loss, chem_pred_attachment_acc, 
                        chem_pred_root_bond_loss, chem_pred_root_bond_acc,
                        ms_pred_motif_loss, ms_pred_motif_acc,
                        ms_pred_attachment_loss, ms_pred_attachment_acc,
                        ms_pred_root_bond_loss, ms_pred_root_bond_acc,
                        motif_res, attached_motif_res, connect_res,
                        vocab_motif_fp_res, vocab_attached_motif_fp_res, vocab_connect_fp_res,
                      ):
        loss_list = {
            'chem_KL': chem_kl_divergence_loss,
            'ms_KL': ms_kl_divergence_loss,
            'chem_pred_motif': chem_pred_motif_loss,
            'ms_pred_motif': ms_pred_motif_loss,
            'chem_pred_att': chem_pred_attachment_loss,
            'ms_pred_att': ms_pred_attachment_loss,
            'chem_pred_root_bond': chem_pred_root_bond_loss,
            'ms_pred_root_bond': ms_pred_root_bond_loss,
            'motif_fp': motif_res[0],
            'att_motif_fp': attached_motif_res[0],
            'connect_fp': connect_res[0],
            'v_motif_fp': vocab_motif_fp_res[0],
            'v_att_motif_fp': vocab_attached_motif_fp_res[0],
            'v_connect_fp': vocab_connect_fp_res[0],
            # 'latent_fp': latent_fp_loss,
        }
        acc_list = {
            'chem_pred_motif': chem_pred_motif_acc.item(),
            'ms_pred_motif': ms_pred_motif_acc.item(),
            'chem_pred_att': chem_pred_attachment_acc.item(),
            'ms_pred_att': ms_pred_attachment_acc.item(),
            'chem_pred_root_bond': chem_pred_root_bond_acc.item(),
            'ms_pred_root_bond': ms_pred_root_bond_acc.item(),
            'motif_fp': motif_res[1].item(),
            'att_motif_fp': attached_motif_res[1].item(),
            'connect_fp': connect_res[1].item(),
            'v_motif_fp': vocab_motif_fp_res[1].item(),
            'v_att_motif_fp': vocab_attached_motif_fp_res[1].item(),
            'v_connect_fp': vocab_connect_fp_res[1].item(),
            # 'latent_fp': latent_fp_acc.item(),
        }
        target_data = {
            'chem_KL': {'loss': chem_kl_divergence_loss.item(), 'accuracy': None, 'criterion': 'KL Divergence'},
            'ms_KL': {'loss': ms_kl_divergence_loss.item(), 'accuracy': None, 'criterion': 'KL Divergence'},
            'chem_pred_motif': {'loss': chem_pred_motif_loss.item(), 'accuracy': chem_pred_motif_acc.item(), 'criterion': get_criterion_name(self.pred_motif_loss_fn)},
            'ms_pred_motif': {'loss': ms_pred_motif_loss.item(), 'accuracy': ms_pred_motif_acc.item(), 'criterion': get_criterion_name(self.pred_motif_loss_fn)},
            'chem_pred_att': {'loss': chem_pred_attachment_loss.item(), 'accuracy': chem_pred_attachment_acc.item(), 'criterion': get_criterion_name(self.pred_attachment_loss_fn)},
            'ms_pred_att': {'loss': ms_pred_attachment_loss.item(), 'accuracy': ms_pred_attachment_acc.item(), 'criterion': get_criterion_name(self.pred_attachment_loss_fn)},
            'chem_pred_root_bond': {'loss': chem_pred_root_bond_loss.item(), 'accuracy': chem_pred_root_bond_acc.item(), 'criterion': get_criterion_name(self.pred_root_bond_pos_loss_fn)},
            'ms_pred_root_bond': {'loss': ms_pred_root_bond_loss.item(), 'accuracy': ms_pred_root_bond_acc.item(), 'criterion': get_criterion_name(self.pred_root_bond_pos_loss_fn)},
            'motif_fp': {'loss': motif_res[0].item(), 'accuracy': motif_res[1].item(), 'criterion': get_criterion_name(self.frag_embedding.fp_loss_fn)},
            'att_motif_fp': {'loss': attached_motif_res[0].item(), 'accuracy': attached_motif_res[1].item(), 'criterion': get_criterion_name(self.frag_embedding.fp_loss_fn)},
            'connect_fp': {'loss': connect_res[0].item(), 'accuracy': connect_res[1].item(), 'criterion': get_criterion_name(self.frag_embedding.fp_loss_fn)},
            'v_motif_fp': {'loss': vocab_motif_fp_res[0].item(), 'accuracy': vocab_motif_fp_res[1].item(), 'criterion': get_criterion_name(self.frag_embedding.fp_loss_fn)},
            'v_att_motif_fp': {'loss': vocab_attached_motif_fp_res[0].item(), 'accuracy': vocab_attached_motif_fp_res[1].item(), 'criterion': get_criterion_name(self.frag_embedding.fp_loss_fn)},
            'v_connect_fp': {'loss': vocab_connect_fp_res[0].item(), 'accuracy': vocab_connect_fp_res[1].item(), 'criterion': get_criterion_name(self.frag_embedding.fp_loss_fn)},
            # 'latent_fp': {'loss': latent_fp_loss.item(), 'accuracy': latent_fp_acc.item(), 'criterion': get_criterion_name(self.fp_loss_fn)},
        }

        return loss_list, acc_list, target_data

    def loss_res_val(self,
                        chem_kl_divergence_loss, ms_kl_divergence_loss,
                        chem_pred_motif_loss, chem_pred_motif_acc, 
                        chem_pred_attachment_loss, chem_pred_attachment_acc, 
                        chem_pred_root_bond_loss, chem_pred_root_bond_acc,
                        ms_pred_motif_loss, ms_pred_motif_acc,
                        ms_pred_attachment_loss, ms_pred_attachment_acc,
                        ms_pred_root_bond_loss, ms_pred_root_bond_acc,
                      ):
        loss_list = {
            'chem_KL': chem_kl_divergence_loss,
            'ms_KL': ms_kl_divergence_loss,
            'chem_pred_motif': chem_pred_motif_loss,
            'ms_pred_motif': ms_pred_motif_loss,
            'chem_pred_att': chem_pred_attachment_loss,
            'ms_pred_att': ms_pred_attachment_loss,
            'chem_pred_root_bond': chem_pred_root_bond_loss,
            'ms_pred_root_bond': ms_pred_root_bond_loss,
        }
        acc_list = {
            'chem_pred_motif': chem_pred_motif_acc.item(),
            'ms_pred_motif': ms_pred_motif_acc.item(),
            'chem_pred_att': chem_pred_attachment_acc.item(),
            'ms_pred_att': ms_pred_attachment_acc.item(),
            'chem_pred_root_bond': chem_pred_root_bond_acc.item(),
            'ms_pred_root_bond': ms_pred_root_bond_acc.item(),
        }
        target_data = {
            'chem_KL': {'loss': chem_kl_divergence_loss.item(), 'accuracy': None, 'criterion': 'KL Divergence'},
            'ms_KL': {'loss': ms_kl_divergence_loss.item(), 'accuracy': None, 'criterion': 'KL Divergence'},
            'chem_pred_motif': {'loss': chem_pred_motif_loss.item(), 'accuracy': chem_pred_motif_acc.item(), 'criterion': get_criterion_name(self.pred_motif_loss_fn)},
            'ms_pred_motif': {'loss': ms_pred_motif_loss.item(), 'accuracy': ms_pred_motif_acc.item(), 'criterion': get_criterion_name(self.pred_motif_loss_fn)},
            'chem_pred_att': {'loss': chem_pred_attachment_loss.item(), 'accuracy': chem_pred_attachment_acc.item(), 'criterion': get_criterion_name(self.pred_attachment_loss_fn)},
            'ms_pred_att': {'loss': ms_pred_attachment_loss.item(), 'accuracy': ms_pred_attachment_acc.item(), 'criterion': get_criterion_name(self.pred_attachment_loss_fn)},
            'chem_pred_root_bond': {'loss': chem_pred_root_bond_loss.item(), 'accuracy': chem_pred_root_bond_acc.item(), 'criterion': get_criterion_name(self.pred_root_bond_pos_loss_fn)},
            'ms_pred_root_bond': {'loss': ms_pred_root_bond_loss.item(), 'accuracy': ms_pred_root_bond_acc.item(), 'criterion': get_criterion_name(self.pred_root_bond_pos_loss_fn)},
        }

        return loss_list, acc_list, target_data


    def get_config_param(self):
        """
        Get model configuration automatically.

        Returns:
            dict: Model configuration.
        """
        # Get all constructor parameter names dynamically
        constructor_params = inspect.signature(self.__init__).parameters
        ignore_config_keys = ['self', 'vocab_data']
        config_keys = [param for param in constructor_params if param not in ignore_config_keys]

        # Extract only the required parameters from instance attributes
        config = {key: getattr(self, key) for key in config_keys if hasattr(self, key)}
        
        return config
    
    @staticmethod
    def from_config_param(config_param):
        """
        Create model from configuration parameters.

        Args:
            config_param (dict): Model configuration parameters.

        Returns:
            Ms2z: Model instance.
        """
        return Ms2z(**config_param)

def get_criterion_name(criterion):
    """
    Get the name of a criterion (function or class).

    Args:
        criterion: The criterion object (function or class).

    Returns:
        str: The name of the criterion.
    """
    if criterion.__class__.__name__ == 'function':
        return criterion.__name__ 
    else:
        return criterion.__class__.__name__ 

class LatentSampler(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        LatentSampler: Encodes input into a latent space using mean and log variance.

        Args:
            input_dim (int): Dimension of the input features.
            latent_dim (int): Dimension of the latent space.
        """
        super(LatentSampler, self).__init__()

        # Linear layers for mean and log variance
        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_log_var = nn.Linear(input_dim, latent_dim)

    @staticmethod
    def reparameterize(mean, log_var):
        """
        Reparameterization trick to sample z.

        Args:
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.

        Returns:
            torch.Tensor: Sampled latent variable z.
        """
        std = torch.exp(0.5 * log_var)  # Standard deviation
        epsilon = torch.randn_like(std)  # Random noise
        return mean + epsilon * std

    def forward(self, encoder_output):
        """
        Forward pass for LatentSampler.

        Args:
            encoder_output (torch.Tensor): Output from the encoder, shape (batch_size, input_dim).

        Returns:
            z (torch.Tensor): Sampled latent variable, shape (batch_size, latent_dim).
            mean (torch.Tensor): Mean of the latent variable, shape (batch_size, latent_dim).
            log_var (torch.Tensor): Log variance of the latent variable, shape (batch_size, latent_dim).
        """
        mean = self.fc_mean(encoder_output)  # Compute mean
        log_var = self.fc_log_var(encoder_output)  # Compute log variance

        z = LatentSampler.reparameterize(mean, log_var)  # Sample z using the reparameterization trick

        return z, mean, log_var
    
    def calc_kl_divergence(self, mean, log_var):
        """
        Calculate the KL divergence with optional target mean and variance.

        Args:
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.

        Returns:
            torch.Tensor: KL divergence.
        """
        kl_divergence = - 0.5 * torch.sum(
            1 + log_var - mean**2 - log_var.exp(),
            dim=1)
        
        return kl_divergence.mean()

    @staticmethod
    def bhattacharyya_coefficient(mean1, log_var1, mean2, log_var2):
        """
        Compute Bhattacharyya coefficient for two diagonal Gaussian distributions.

        Args:
            mean1 (torch.Tensor): Mean vectors of the first distribution [datasize, dim]
            log_var1 (torch.Tensor): Log variance vectors of the first distribution [datasize, dim]
            mean2 (torch.Tensor): Mean vectors of the second distribution [datasize, dim]
            log_var2 (torch.Tensor): Log variance vectors of the second distribution [datasize, dim]

        Returns:
            torch.Tensor: Bhattacharyya coefficient [datasize]
        """
        var1 = torch.exp(log_var1)  # Convert log variance to variance
        var2 = torch.exp(log_var2)
        var_avg = 0.5 * (var1 + var2)  # Average variance

        mean_diff = mean1 - mean2  # Mean difference

        # Bhattacharyya coefficient calculation
        exp_term = torch.exp(-0.25 * torch.sum(mean_diff ** 2 / var_avg, dim=1))
        coef_term = torch.prod(torch.sqrt(2 * var1 * var2 / (var1 + var2)), dim=1)

        return exp_term * coef_term


class ModelChemInput(NamedTuple):
    token: torch.Tensor
    order: torch.Tensor
    mask: torch.Tensor

    def __len__(self):
        return self.token.size(0)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return ModelChemInput(
                token=self.token[idx].unsqueeze(0),
                order=self.order[idx].unsqueeze(0),
                mask=self.mask[idx].unsqueeze(0)
            )
        elif isinstance(idx, slice) or isinstance(idx, torch.Tensor) or isinstance(idx, list):
            return ModelChemInput(
                token=self.token[idx],
                order=self.order[idx],
                mask=self.mask[idx]
            )
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def to(self, device):
        return ModelChemInput(
            token=self.token.to(device),
            order=self.order.to(device),
            mask=self.mask.to(device)
        )
    def to_tuple(self):
        return (self.token, self.order, self.mask)
    
    @staticmethod
    def from_tuple(data):
        return ModelChemInput(*data)
    
    @staticmethod
    def cat(*inputs):
        token_cat = torch.cat([i.token for i in inputs], dim=0)
        order_cat = torch.cat([i.order for i in inputs], dim=0)
        mask_cat = torch.cat([i.mask for i in inputs], dim=0)
        return ModelChemInput(token=token_cat, order=order_cat, mask=mask_cat)

    
class ModelMsInput(NamedTuple):
    precursor: torch.Tensor
    fragment: torch.Tensor
    fragment_mask: torch.Tensor

    def __len__(self):
        return self.precursor.size(0)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return ModelMsInput(
                precursor=self.precursor[idx],
                fragment=self.fragment[idx],
                fragment_mask=self.fragment_mask[idx],
            )
        elif isinstance(idx, slice) or isinstance(idx, torch.Tensor) or isinstance(idx, list):
            return ModelMsInput(
                precursor=self.precursor[idx],
                fragment=self.fragment[idx],
                fragment_mask=self.fragment_mask[idx],
            )
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def to(self, device):
        return ModelMsInput(
            precursor=self.precursor.to(device),
            fragment=self.fragment.to(device),
            fragment_mask=self.fragment_mask.to(device),
        )
    
    def to_tuple(self):
        return (self.precursor, self.fragment, self.fragment_mask)
    
    @staticmethod
    def from_tuple(data):
        return ModelMsInput(*data)
    
class ModelDecoderInput(NamedTuple):
    node_embed: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    tgt_mask: torch.Tensor
    edge_mask: torch.Tensor

    def __len__(self):
        return self.token.size(0)

    def to(self, device):
        return ModelDecoderInput(
            node_embed=self.node_embed.to(device),
            edge_index=self.edge_index.to(device),
            tgt_mask=self.tgt_mask.to(device),
            edge_mask=self.edge_mask.to(device),
        )
    
    def to_tuple(self):
        return (self.node_embed, self.edge_index, self.tgt_mask, self.edge_mask)
    
    @staticmethod
    def from_tuple(data):
        return ModelDecoderInput(*data)