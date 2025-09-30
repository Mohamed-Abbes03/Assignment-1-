import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

# ---------------------------
# Multi-Head Attention
# ---------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, traces=None, prefix=""):
        # q, k, v shapes: (batch, seq_len, d_model)
        bs = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)

        # Linear projections
        Q = self.q_linear(q).view(bs, q_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(bs, k_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(bs, k_len, self.num_heads, self.d_k).transpose(1, 2)
        # Now Q, K, V shapes: (batch, num_heads, seq_len, d_k)  (note: seq_len differs for Q vs K/V)

        if traces is not None:
            traces[f"{prefix} - Q"] = Q.clone().detach()
            traces[f"{prefix} - K"] = K.clone().detach()
            traces[f"{prefix} - V"] = V.clone().detach()
            # Multi-head split (pack)
            traces[f"{prefix} - Multi-Head Split"] = [Q.clone().detach(), K.clone().detach(), V.clone().detach()]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: (batch, num_heads, q_len, k_len)
        if traces is not None:
            traces[f"{prefix} - Attention Scores Before Softmax"] = scores.clone().detach()

        if mask is not None:
            # mask expected to broadcast to (batch, 1, q_len, k_len) or be already (batch, 1, q_len, k_len)
            if traces is not None:
                traces["25. Mask tensor (as passed)"] = mask.clone().detach()
            scores = scores.masked_fill(mask == 0, float('-1e9'))
            if traces is not None:
                traces[f"{prefix} - Attention Scores After Mask (Before Softmax)"] = scores.clone().detach()

        attn = F.softmax(scores, dim=-1)
        if traces is not None:
            traces[f"{prefix} - Attention Scores After Softmax"] = attn.clone().detach()

        context = torch.matmul(attn, V)  # (batch, num_heads, q_len, d_k)
        concat = context.transpose(1, 2).contiguous().view(bs, q_len, self.num_heads * self.d_k)
        if traces is not None:
            traces[f"{prefix} - Multi-Head Concatenated"] = concat.clone().detach()

        output = self.out(concat)  # (batch, q_len, d_model)
        if traces is not None:
            traces[f"{prefix} - Attention Output"] = output.clone().detach()

        return output

# ---------------------------
# Feed Forward
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x, traces=None, prefix=""):
        # x shape: (batch, seq_len, d_model)
        if traces is not None:
            traces[f"{prefix} - FF Input"] = x.clone().detach()
        inter = self.linear1(x)
        if traces is not None:
            traces[f"{prefix} - FF Linear1"] = inter.clone().detach()
        inter_act = F.relu(inter)
        out = self.linear2(inter_act)
        if traces is not None:
            traces[f"{prefix} - FF Linear2"] = out.clone().detach()
        return out

# ---------------------------
# Encoder Layer
# ---------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, traces=None):
        # x shape: (batch, seq_len, d_model)
        if traces is not None:
            traces["6. Encoder block input tensor"] = x.clone().detach()

        # Self-attention
        attn_out = self.mha(x, x, x, mask=mask, traces=traces, prefix="Encoder Self-Attention")
        # Map MHA traces to required snapshot keys:
        if traces is not None:
            # Q/K/V for snapshot 7/8/9
            traces["7. Self-attention queries (Q)"] = traces.get("Encoder Self-Attention - Q").clone().detach()
            traces["8. Self-attention keys (K)"] = traces.get("Encoder Self-Attention - K").clone().detach()
            traces["9. Self-attention values (V)"] = traces.get("Encoder Self-Attention - V").clone().detach()
            # scores before softmax -> 10
            traces["10. Attention score matrix before softmax"] = traces.get("Encoder Self-Attention - Attention Scores Before Softmax").clone().detach()
            # scores after softmax -> 11
            traces["11. Attention score matrix after softmax"] = traces.get("Encoder Self-Attention - Attention Scores After Softmax").clone().detach()
            # multi-head split -> 12
            traces["12. Multi-head split (Q/K/V split)"] = traces.get("Encoder Self-Attention - Multi-Head Split")
            # concatenated output -> 13
            traces["13. Multi-head attention output after concatenation"] = traces.get("Encoder Self-Attention - Multi-Head Concatenated").clone().detach()
            # attention output -> will be covered as part of residuals etc.

        res1 = x + attn_out
        if traces is not None:
            traces["14. Residual connection tensors"] = res1.clone().detach()

        norm1 = self.norm1(res1)
        if traces is not None:
            traces["15. Layer normalization output"] = norm1.clone().detach()

        # Feed-forward
        if traces is not None:
            traces["16. Feed-forward input"] = norm1.clone().detach()
        ff_out = self.ff(norm1, traces, prefix="Encoder FFN")
        if traces is not None:
            traces["17. Feed-forward first linear layer output"] = traces.get("Encoder FFN - FF Linear1").clone().detach()
            traces["18. Feed-forward second linear layer output"] = traces.get("Encoder FFN - FF Linear2").clone().detach()

        res2 = norm1 + ff_out
        if traces is not None:
            traces["14b. Residual (post-FF)"] = res2.clone().detach()  # optional extra snapshot
        norm2 = self.norm2(res2)
        if traces is not None:
            traces["19. Encoder block final output tensor"] = norm2.clone().detach()

        return norm2

# ---------------------------
# Decoder Layer
# ---------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None, traces=None):
        # x shape: (batch, tgt_len, d_model)
        if traces is not None:
            traces["20. Decoder block input tensor"] = x.clone().detach()

        # Masked self-attention (causal)
        self_attn_out = self.self_attn(x, x, x, mask=tgt_mask, traces=traces, prefix="Decoder Masked Self-Attention")
        # Map masked self-attention traces to required snapshots
        if traces is not None:
            traces["21. Masked self-attention queries (Q)"] = traces.get("Decoder Masked Self-Attention - Q").clone().detach()
            traces["22. Masked self-attention keys (K)"] = traces.get("Decoder Masked Self-Attention - K").clone().detach()
            traces["23. Masked self-attention values (V)"] = traces.get("Decoder Masked Self-Attention - V").clone().detach()
            # Scores before mask (we saved raw scores before applying mask in MultiHeadAttention maybe not available separately here).
            # We saved "Decoder Masked Self-Attention - Attention Scores Before Softmax"
            traces["24. Masked attention scores before mask"] = traces.get("Decoder Masked Self-Attention - Attention Scores Before Softmax").clone().detach()
            # Mask tensor snapshot -> 25 (we will also set it on call site)
            # After mask + softmax:
            # After mask but before softmax was saved as "Decoder Masked Self-Attention - Attention Scores After Mask (Before Softmax)"
            traces["26. Masked attention scores after mask + softmax"] = traces.get("Decoder Masked Self-Attention - Attention Scores After Softmax").clone().detach()
            traces["27. Masked self-attention multi-head split"] = traces.get("Decoder Masked Self-Attention - Multi-Head Split")
            traces["28. Masked self-attention multi-head concatenated output"] = traces.get("Decoder Masked Self-Attention - Multi-Head Concatenated").clone().detach()

        res1 = x + self_attn_out
        norm1 = self.norm1(res1)
        if traces is not None:
            traces["29. Residual + normalization after masked self-attention"] = norm1.clone().detach()

        # Cross-attention: queries come from decoder (norm1), keys/values from encoder output
        cross_attn_out = self.cross_attn(norm1, enc_out, enc_out, mask=src_mask, traces=traces, prefix="Decoder Cross-Attention")
        if traces is not None:
            traces["30. Cross-attention queries (from decoder)"] = traces.get("Decoder Cross-Attention - Q").clone().detach()
            traces["31. Cross-attention keys (from encoder)"] = traces.get("Decoder Cross-Attention - K").clone().detach()
            traces["32. Cross-attention values (from encoder)"] = traces.get("Decoder Cross-Attention - V").clone().detach()
            traces["33. Cross-attention score matrix before softmax"] = traces.get("Decoder Cross-Attention - Attention Scores Before Softmax").clone().detach()
            traces["34. Cross-attention score matrix after softmax"] = traces.get("Decoder Cross-Attention - Attention Scores After Softmax").clone().detach()
            traces["35. Cross-attention output after concatenation"] = traces.get("Decoder Cross-Attention - Multi-Head Concatenated").clone().detach()

        res2 = norm1 + cross_attn_out
        norm2 = self.norm2(res2)
        if traces is not None:
            traces["36. Residual + normalization after cross-attention"] = norm2.clone().detach()

        # Feed-forward in decoder
        if traces is not None:
            traces["37. Decoder feed-forward input"] = norm2.clone().detach()
        ff_out = self.ff(norm2, traces, prefix="Decoder FFN")
        if traces is not None:
            traces["38. Feed-forward first linear layer output"] = traces.get("Decoder FFN - FF Linear1").clone().detach()
            traces["39. Feed-forward second linear layer output"] = traces.get("Decoder FFN - FF Linear2").clone().detach()

        res3 = norm2 + ff_out
        norm3 = self.norm3(res3)
        if traces is not None:
            traces["40. Decoder block final output tensor"] = norm3.clone().detach()

        return norm3

# ---------------------------
# Transformer
# ---------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, N=2, num_heads=4, d_ff=512, max_len=512, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(N)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(N)])
        self.out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    @staticmethod
    def generate_causal_mask(batch_size, tgt_len, device):
        # Lower-triangular ones allow attention; zeros block future positions.
        base = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=device))
        # Reshape to (batch, 1, tgt_len, tgt_len) so it can broadcast over heads and queries
        return base.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)

    def forward(self, src, tgt):
        traces = {}

        # 1. Raw input tokens
        traces["1. Raw input tokens"] = src.clone().detach()
        traces["2. Target tokens"] = tgt.clone().detach()

        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt.size(1)

        src_mask = torch.ones((batch_size, 1, 1, src_len), device=src.device, dtype=torch.uint8)  # allow all src positions
        # causal tgt mask
        tgt_mask = self.generate_causal_mask(batch_size, tgt_len, device=tgt.device)  # (batch,1,tgt_len,tgt_len)

        # store mask snapshot (25)
        traces["25. Mask tensor"] = tgt_mask.clone().detach()

        # Embeddings
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)  # (batch, src_len, d_model)
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)  # (batch, tgt_len, d_model)

        traces["3. Embedding weight matrix (slice)"] = self.src_embed.weight[:5, :5].clone().detach()
        traces["4. Input embeddings after lookup"] = src_emb.clone().detach()
        traces["5. Embeddings after adding positional encoding"] = self.pos_enc(src_emb).clone().detach()

        # Encoder
        enc_out = self.pos_enc(src_emb)
        for layer in self.enc_layers:
            enc_out = layer(enc_out, mask=src_mask, traces=traces)

        # Decoder (prepare input)
        dec_in = self.pos_enc(tgt_emb)
        # Save decoder input snapshot (20)
        traces["20. Decoder block input tensor"] = dec_in.clone().detach()

        for layer in self.dec_layers:
            dec_in = layer(dec_in, enc_out, src_mask=src_mask, tgt_mask=tgt_mask, traces=traces)

        traces["41. Decoder final sequence output (before projection)"] = dec_in.clone().detach()

        logits = self.out(dec_in)  # (batch, tgt_len, vocab_size)
        traces["42. Logits after final linear projection"] = logits.clone().detach()
        traces["43. Logits slice (first few values for one token)"] = logits[0, 0, :5].clone().detach()

        # Ensure the mandatory keys 6..40 exist (some are created inside layers).  If any missing, fill placeholders to avoid KeyError when students expect them:
        required_keys = [
            "6. Encoder block input tensor", "7. Self-attention queries (Q)", "8. Self-attention keys (K)", "9. Self-attention values (V)",
            "10. Attention score matrix before softmax", "11. Attention score matrix after softmax", "12. Multi-head split (Q/K/V split)",
            "13. Multi-head attention output after concatenation", "14. Residual connection tensors", "15. Layer normalization output",
            "16. Feed-forward input", "17. Feed-forward first linear layer output", "18. Feed-forward second linear layer output", "19. Encoder block final output tensor",
            "21. Masked self-attention queries (Q)", "22. Masked self-attention keys (K)", "23. Masked self-attention values (V)",
            "24. Masked attention scores before mask", "26. Masked attention scores after mask + softmax",
            "27. Masked self-attention multi-head split", "28. Masked self-attention multi-head concatenated output",
            "29. Residual + normalization after masked self-attention", "30. Cross-attention queries (from decoder)",
            "31. Cross-attention keys (from encoder)", "32. Cross-attention values (from encoder)",
            "33. Cross-attention score matrix before softmax", "34. Cross-attention score matrix after softmax",
            "35. Cross-attention output after concatenation", "36. Residual + normalization after cross-attention",
            "37. Decoder feed-forward input", "38. Feed-forward first linear layer output", "39. Feed-forward second linear layer output", "40. Decoder block final output tensor"
        ]
        for k in required_keys:
            if k not in traces:
                # Insert small zero-shaped placeholders so students can still inspect shape and replace with real snapshots when they run the debugger
                traces[k] = torch.tensor([])

        return logits, traces


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # build a small vocabulary including words used in src/tgt sentences
    stoi = {
        "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
        "The": 4, "old": 5, "wizard": 6, "spoke": 7, "in": 8, "riddles": 9, "to": 10, "Dobby": 11,
        "He": 12, "answered": 13, "with": 14, "courage": 15, "and": 16, "wisdom": 17,
        ".": 18
    }
    itos = {i: s for s, i in stoi.items()}

    # choose source and target of length 6 tokens each (allowed 5-12)
    src_sentence = "The old wizard spoke in riddles to"
    # tokens: ["The","old","wizard","spoke","in","riddles"] (we drop trailing "to" to get 6 tokens)
    src_words = ["The", "old", "wizard", "spoke", "in", "riddles"]
    tgt_sentence = "He answered with courage and wisdom"
    tgt_words = ["He", "answered", "with", "courage", "and", "wisdom"]

    # Create token ID tensors
    src_ids = torch.tensor([[stoi.get(w, stoi["<unk>"]) for w in src_words]], dtype=torch.long)
    tgt_ids = torch.tensor([[stoi.get(w, stoi["<unk>"]) for w in tgt_words]], dtype=torch.long)

    # Instantiate model with required sizes: d_model=128, N=2, num_heads=4
    model = Transformer(len(stoi), len(stoi), d_model=128, N=2, num_heads=4, d_ff=512)
    model.eval()

    logits, traces = model(src_ids, tgt_ids)

    # drop into debugger so students can step through and capture snapshots (no prints)
    breakpoint()
