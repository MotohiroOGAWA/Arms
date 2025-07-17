import unittest
import torch
import os
from model.utils.transformer.MultiHeadAttention import MultiHeadAttentionBlock

class TestMultiHeadAttentionBlock(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 3
        self.embed_dim = 4
        self.num_heads = 2
        self.dropout = 0.0 

        self.model = MultiHeadAttentionBlock(
            embed_dim=self.embed_dim,
            h=self.num_heads,
            dropout=self.dropout
        )

        self.q = torch.ones(self.batch_size, self.seq_len, self.embed_dim)
        self.k = torch.arange(1, self.batch_size * self.seq_len * self.embed_dim + 1).float().reshape(
            self.batch_size, self.seq_len, self.embed_dim
        )
        self.v = torch.arange(1, self.batch_size * self.seq_len * self.embed_dim + 1).float().reshape(
            self.batch_size, self.seq_len, self.embed_dim
        )

        with torch.no_grad():
            d = self.embed_dim  # for example, 4

            # w_q: Row-major increasing pattern (e.g., [0,1,2,3], [4,5,6,7], ...)
            w_q_matrix = torch.arange(d * d).view(d, d)
            self.model.w_q.weight.copy_(w_q_matrix)

            # w_k: Column-major increasing pattern (e.g., [[0,4,8,12], [1,5,9,13], ...])
            w_k_matrix = torch.arange(d * d).view(d, d).t().contiguous()
            self.model.w_k.weight.copy_(w_k_matrix)

            # w_v: Checkerboard pattern (e.g., alternating 1s and 0s)
            w_v_matrix = torch.zeros(d, d, dtype=torch.int)
            for i in range(d):
                for j in range(d):
                    w_v_matrix[i, j] = (i + j) % 2
            self.model.w_v.weight.copy_(w_v_matrix)

            # w_o: Diagonal filled with 2, off-diagonal with 1
            w_o_matrix = torch.ones(d, d, dtype=torch.int)
            for i in range(d):
                w_o_matrix[i, i] = 2
            self.model.w_o.weight.copy_(w_o_matrix)


    def test_deterministic_output(self):
        key_mask = torch.tensor([[False, False, False]])  # No masking
        out = self.model(self.q, self.k, self.v, key_mask=key_mask)

        expected = torch.tensor([[[106., 104., 106., 104.],
                                  [106., 104., 106., 104.],
                                  [106., 104., 106., 104.]],

                                 [[226., 224., 226., 224.],
                                  [226., 224., 226., 224.],
                                  [226., 224., 226., 224.]]])

        self.assertTrue(torch.allclose(out, expected, atol=1e-2), f"Output was {out}")

    def test_broadcastable_high_dim_mask(self):
        key_mask = torch.tensor([[False, False, False]]).unsqueeze(0).repeat(3, 1, 1)
        causal_mask = torch.tensor([[False, False, False]]).unsqueeze(0).repeat(3, 1, 1)
        q = self.q.unsqueeze(0).repeat(3, 1, 1, 1)
        k = self.k.unsqueeze(0).repeat(3, 1, 1, 1)
        v = self.v.unsqueeze(0).repeat(3, 1, 1, 1)
        out = self.model(q, k, v, causal_mask=causal_mask, key_mask=key_mask)

        expected = torch.tensor([[[106., 104., 106., 104.],
                                  [106., 104., 106., 104.],
                                  [106., 104., 106., 104.]],

                                 [[226., 224., 226., 224.],
                                  [226., 224., 226., 224.],
                                  [226., 224., 226., 224.]]]).unsqueeze(0).repeat(3, 1, 1, 1)
        
        self.assertTrue(torch.allclose(out, expected, atol=1e-2), f"Output was {out}")

    def test_mask_applied_effectively(self):
        # Mask the second key (index 1)
        key_mask = torch.tensor([[False, False, True]])
        out = self.model(self.q, self.k, self.v, key_mask=key_mask)

        # Output should differ because the 2nd key/value is masked
        expected = torch.tensor([[[ 66.,  64.,  66.,  64.],
                                  [ 66.,  64.,  66.,  64.],
                                  [ 66.,  64.,  66.,  64.]],

                                 [[186., 184., 186., 184.],
                                  [186., 184., 186., 184.],
                                  [186., 184., 186., 184.]]])  # This is computed manually or verified from unmasked version

        self.assertTrue(torch.allclose(out, expected, atol=1e-2), f"Output with masked key was {out}")

    def test_export_to_onnx(self):
        """
        This test exports the model to ONNX format for visualization/debugging.
        It does not assert anything and is skipped unless explicitly run.
        """
        import torch.onnx
        # Check if the test is run with ONNX export flag
        # if not getattr(self, "_run_onnx_export", False):
        #     self.skipTest("Skip ONNX export unless _run_onnx_export flag is set")

        self.model.eval()

        # Export the model with dummy input
        output_dir = "tests/data/model/utils/transformer"
        os.makedirs(output_dir, exist_ok=True)
        torch.onnx.export(
            self.model,
            (self.q, self.k, self.v),
            os.path.join(output_dir, "multihead_attention.onnx"),
            input_names=["q", "k", "v"],
            output_names=["output"],
            dynamic_axes={
                "q": {0: "batch"}, "k": {0: "batch"}, "v": {0: "batch"}, "output": {0: "batch"}
            },
            opset_version=11,
            verbose=False
        )
        print("Exported ONNX model to multihead_attention.onnx")


if __name__ == "__main__":
    unittest.main()

        
