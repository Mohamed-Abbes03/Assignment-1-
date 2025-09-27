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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

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
        bs = q.size(0)

        Q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)

        if traces is not None:
            traces[f"{prefix} - Q"] = Q.clone().detach()
            traces[f"{prefix} - K"] = K.clone().detach()
            traces[f"{prefix} - V"] = V.clone().detach()
            traces[f"{prefix} - Multi-Head Split"] = [Q.clone().detach(),
                                                      K.clone().detach(),
                                                      V.clone().detach()]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if traces is not None:
            traces[f"{prefix} - Attention Scores Before Softmax"] = scores.clone().detach()

        if mask is not None:
            if traces is not None:
                traces["Mask Tensor"] = mask.clone().detach()
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        if traces is not None:
            traces[f"{prefix} - Attention Scores After Softmax"] = attn.clone().detach()

        context = torch.matmul(attn, V)
        concat = context.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)

        if traces is not None:
            traces[f"{prefix} - Multi-Head Concatenated"] = concat.clone().detach()

        output = self.out(concat)
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
        if traces is not None:
            traces[f"{prefix} - FF Input"] = x.clone().detach()
        inter = self.linear1(x)
        if traces is not None:
            traces[f"{prefix} - FF Linear1"] = inter.clone().detach()
        inter = F.relu(inter)
        out = self.linear2(inter)
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
        if traces is not None:
            traces["6. Encoder Input"] = x.clone().detach()

        attn_out = self.mha(x, x, x, mask, traces, prefix="Encoder Self-Attention")
        res1 = x + attn_out
        if traces is not None:
            traces["14. Encoder Residual"] = res1.clone().detach()

        norm1 = self.norm1(res1)
        if traces is not None:
            traces["15. Encoder LayerNorm1"] = norm1.clone().detach()

        ff_out = self.ff(norm1, traces, prefix="Encoder FFN")
        res2 = norm1 + ff_out
        if traces is not None:
            traces["18. Encoder Residual"] = res2.clone().detach()

        norm2 = self.norm2(res2)
        if traces is not None:
            traces["19. Encoder Output"] = norm2.clone().detach()

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
        if traces is not None:
            traces["20. Decoder Input"] = x.clone().detach()

        self_attn_out = self.self_attn(x, x, x, tgt_mask, traces, prefix="Decoder Masked Self-Attention")
        res1 = x + self_attn_out
        norm1 = self.norm1(res1)
        if traces is not None:
            traces["29. Decoder Residual+Norm1"] = norm1.clone().detach()

        cross_attn_out = self.cross_attn(norm1, enc_out, enc_out, src_mask, traces, prefix="Decoder Cross-Attention")
        res2 = norm1 + cross_attn_out
        norm2 = self.norm2(res2)
        if traces is not None:
            traces["36. Decoder Residual+Norm2"] = norm2.clone().detach()

        ff_out = self.ff(norm2, traces, prefix="Decoder FFN")
        res3 = norm2 + ff_out
        norm3 = self.norm3(res3)
        if traces is not None:
            traces["40. Decoder Output"] = norm3.clone().detach()

        return norm3

# ---------------------------
# Transformer
# ---------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=16, N=1, num_heads=2, d_ff=32, max_len=512):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(N)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(N)])
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        traces = {}

        traces["1. Raw Source IDs"] = src.clone().detach()
        traces["2. Raw Target IDs"] = tgt.clone().detach()

        src_mask = torch.ones(src.size(0), 1, 1, src.size(1), device=src.device)
        tgt_mask = torch.ones(tgt.size(0), 1, tgt.size(1), tgt.size(1), device=tgt.device)
        traces["Mask Tensor"] = src_mask.clone().detach()

        src_emb = self.src_embed(src)
        tgt_emb = self.tgt_embed(tgt)
        traces["3. Embedding Weights (src)"] = self.src_embed.weight[:5, :5].clone().detach()
        traces["4. Source Embeddings"] = src_emb.clone().detach()
        traces["5. Source Embeddings + PosEnc"] = self.pos_enc(src_emb).clone().detach()

        enc_out = self.pos_enc(src_emb)
        for layer in self.enc_layers:
            enc_out = layer(enc_out, mask=src_mask, traces=traces)

        tgt_out = self.pos_enc(tgt_emb)
        traces["41. Decoder Final Input Before Layers"] = tgt_out.clone().detach()

        for layer in self.dec_layers:
            tgt_out = layer(tgt_out, enc_out, src_mask=src_mask, tgt_mask=tgt_mask, traces=traces)

        traces["41. Decoder Final Sequence Output"] = tgt_out.clone().detach()

        logits = self.out(tgt_out)
        traces["42. Logits"] = logits.clone().detach()
        traces["43. Logits Slice"] = logits[0, 0, :5].clone().detach()

        return logits, traces


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    stoi = {"<pad>":0,"<s>":1,"</s>":2,"<unk>":3,"Dobby":4,"is":5,"free":6,"He":7,"runs":8}
    itos = {i:s for s,i in stoi.items()}

    src_sentence = "Dobby is free"
    tgt_sentence = "He runs"

    src_ids = torch.tensor([[stoi[w] for w in src_sentence.split()]])
    tgt_ids = torch.tensor([[stoi[w] for w in tgt_sentence.split()]])

    model = Transformer(len(stoi), len(stoi))
    logits, traces = model(src_ids, tgt_ids)

    breakpoint()