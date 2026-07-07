# MA-LMM Inference Codebase Walkthrough

A tensor-level walkthrough of how `scripts/infer.py` runs video captioning using the
Memory-Augmented Large Multimodal Model (MA-LMM) on Ego4D egocentric videos.

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [infer.py — Entry Point](#2-inferpy--entry-point)
3. [Frame Loading & Preprocessing](#3-frame-loading--preprocessing)
4. [Model Architecture Overview](#4-model-architecture-overview)
5. [Visual Encoder (EVA-CLIP-G ViT)](#5-visual-encoder-eva-clip-g-vit)
6. [Visual Memory Bank Accumulation](#6-visual-memory-bank-accumulation)
7. [Visual Memory Bank Compression](#7-visual-memory-bank-compression)
8. [Q-Former: Cross-Attention Deep Dive](#8-q-former-cross-attention-deep-dive)
9. [Query Memory Bank — Per-Layer Self-Attention History](#9-query-memory-bank--per-layer-self-attention-history)
10. [Projection to LLM Space](#10-projection-to-llm-space)
11. [Final LLM Input Assembly](#11-final-llm-input-assembly)
12. [End-to-End Tensor Flow Summary](#12-end-to-end-tensor-flow-summary)
13. [Compression Payoff — Numbers](#13-compression-payoff--numbers)

---

## 1. Repository Layout

```
ma_lmm_baseline/
├── MA-LMM/                          # Cloned upstream repo (provides `lavis` package)
│   └── lavis/
│       └── models/
│           └── blip2_models/
│               ├── blip2_vicuna_instruct.py   # Main model class
│               ├── blip2.py                   # Base class: encoder init, memory_bank_compress
│               └── Qformer.py                 # Q-Former BERT internals
├── scripts/
│   └── infer.py                     # Entry point (this walkthrough)
├── data/
│   ├── annotations/ego4d_10min.json # List of {video_id, n_frames, caption}
│   └── frames/{duration}/{video_id}/frame*.jpg
├── configs/cap_ego4d.yaml           # Reference config (not parsed by infer.py directly)
└── outputs/                         # Per-duration JSON results
```

`infer.py` adds `MA-LMM/` to `sys.path` at runtime so `lavis` can be imported without
a package install.

---

## 2. infer.py — Entry Point

### CLI Arguments

| Argument | Default | Purpose |
|---|---|---|
| `--frame_dir` | required | Root dir with per-video frame folders |
| `--ann_path` | required | JSON annotation file |
| `--output` | required | Where to write results JSON |
| `--memory_bank_length` | `40` | Max frames the memory bank holds before compressing |
| `--num_frames` | `80` | Frames sampled per video |
| `--num_beams` | `5` | Beam search width |
| `--max_len` / `--min_len` | `256` / `10` | Output token length bounds |
| `--prompt` | `"Describe what happens..."` | Text prompt for the LLM |
| `--ckpt_path` | `None` | Fine-tuned checkpoint; omit for zero-shot |

### Model Loading

```python
from lavis.models import load_model_and_preprocess
model, _, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct_malmm",   # registered model name
    model_type="vicuna7b",
    is_eval=True,
    device=args.device,
)
# The three return values are (model, vis_processors, txt_processors).
# Processors are discarded — preprocessing is handled manually in load_video_frames().

# Runtime overrides applied directly on the model object:
model.memory_bank_length = args.memory_bank_length   # 40
model.use_memory_bank    = args.memory_bank_length > 0
model.num_frames         = args.num_frames            # 80
```

### Inference Loop

For each video the script:
1. Loads and samples frames → tensor
2. Runs `model.generate(sample, ...)`
3. Records wall-clock time and GPU peak memory
4. **Writes results to disk after every video** (crash-safe incremental save)

---

## 3. Frame Loading & Preprocessing

```python
def load_video_frames(frame_dir, num_frames, image_size=224):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275,  0.40821073],
                    std= [0.26862954, 0.26130258, 0.27577711]),
    ])
    files = sorted(frame_dir.glob("frame*.jpg"))
    idx = np.linspace(0, len(files)-1, num_frames, dtype=int)  # uniform subsampling
    return torch.stack([transform(Image.open(files[i]).convert("RGB")) for i in idx])
    # output shape: [T=80, C=3, H=224, W=224]
```

**Normalization values** are CLIP-style (same distribution EVA-CLIP-G was pretrained on).

**Temporal subsampling** with `np.linspace`: a 10-minute clip at 30 fps has ~18,000 frames;
`linspace` picks 80 evenly-spaced indices so the model sees the whole duration.

The tensor is then reshaped before being handed to `model.generate()`:

```python
frames = load_video_frames(fdir, num_frames)       # [T=80, C=3, H=224, W=224]
         .permute(1, 0, 2, 3)                       # [C=3, T=80, H=224, W=224]
         .unsqueeze(0)                              # [B=1, C=3, T=80, H=224, W=224]
         .to(device)
```

The model's `generate()` expects batch-first with channels before time: `[B, C, T, H, W]`.

---

## 4. Model Architecture Overview

MA-LMM is **InstructBLIP + Vicuna-7B** with a memory bank extension. Three components:

```
┌──────────────────┐     ┌─────────────────┐     ┌────────────────┐
│  Visual Encoder  │────▶│    Q-Former     │────▶│  Vicuna-7B LLM │
│  EVA-CLIP-G ViT  │     │ (BERT-base w/   │     │  (4096-dim)    │
│  (1408-dim, FROZEN)│   │  cross-attn)    │     │                │
└──────────────────┘     │  + Memory Banks │     └────────────────┘
                         └─────────────────┘
```

- **Visual Encoder**: frozen, processes one frame at a time
- **Q-Former**: 32 learnable query tokens; 12 BERT layers with cross-attention every 2 layers
- **llm_proj**: linear layer bridging Q-Former (768-dim) → LLM (4096-dim)
- **Vicuna-7B**: generates the caption autoregressively

### Two kinds of memory bank (both from the paper)

MA-LMM maintains **13 memory banks** in parallel, all capped at `memory_bank_length` (default 40) and all compressed by the same `memory_bank_compress` routine:

| Bank | Count | Where it lives | Shape | Fed into |
|---|---|---|---|---|
| **Visual memory bank** (`F_t`) | 1 (shared) | `self.visual_memory_bank` on the main model | `[B, T, 257, 1408]` | Q-Former **cross-attention** as `encoder_hidden_states` |
| **Query memory bank** (`Z_t`) | **12** (one per BERT self-attn layer) | `self.query_memory_bank` on each `MBBertSelfAttention` | `[B, T, 32+L_text, 768]` | That layer's own **self-attention** as prepended K/V history |

The visual bank stores *raw* per-frame ViT features; the query banks store the *evolving* query-token hidden states after each layer's transformations. Both are frame-by-frame accumulators — see §6/§7 for the visual bank and §9 for the query banks.

---

## 5. Visual Encoder (EVA-CLIP-G ViT)

**Initialization** (`blip2_vicuna_instruct.py:55-63`):

```python
self.visual_encoder, self.ln_vision = self.init_vision_encoder(
    vit_model="eva_clip_g", img_size=224, ...
)
# visual_encoder = EVA-CLIP-G Vision Transformer
# ln_vision      = LayerNorm(1408)

# Frozen — no gradients, pure feature extractor:
for param in self.visual_encoder.parameters():
    param.requires_grad = False
```

**Called frame-by-frame inside `generate()`** (`blip2_vicuna_instruct.py:381`):

```python
for t in range(T):   # T = 80
    image_embeds = self.ln_vision(
        self.visual_encoder(image[:, :, t, :, :])   # slice one frame: [B, C, H, W]
    ).detach()
    # visual_encoder output: [B, 257, 1408]
    #   257 = 256 patch tokens (14x14 grid from 224px image) + 1 CLS token
    #   1408 = EVA-CLIP-G feature dimension
    # ln_vision: LayerNorm applied per-token, shape unchanged: [B, 257, 1408]
    # .detach(): no gradients flow back through the encoder
```

The ViT is **not** a video transformer — it's a plain image ViT running on one frame.
Video understanding is handled entirely downstream by the memory bank and Q-Former.

**Temporal positional embedding** (`blip2_vicuna_instruct.py:383-388`):

```python
position_ids = torch.tensor([t]).unsqueeze(0).expand(B, -1)  # [B, 1]
image_pe     = self.image_pe(position_ids)                   # [B, 1, 1408]  (learned embedding)
image_embeds = image_embeds + image_pe                       # [B, 257, 1408]
image_embeds = image_embeds.unsqueeze(1)                     # [B, 1, 257, 1408]
```

`self.image_pe` is `nn.Embedding(max_num_frames=120, 1408)`, initialized to zeros.
Adding it encodes *when* in the video this frame occurred.

---

## 6. Visual Memory Bank Accumulation

The visual memory bank grows frame by frame (`blip2_vicuna_instruct.py:385-391`):

```python
# t == 0: initialize
self.visual_memory_bank = image_embeds          # [B, 1,   257, 1408]
self.compression_size   = torch.ones(B, 1, 257) # [B, 1,   257]  tracks merge history

# t > 0: concatenate
self.visual_memory_bank = torch.cat(
    [self.visual_memory_bank, image_embeds], dim=1
)                                               # [B, t+1, 257, 1408]
self.compression_size   = torch.cat(
    [self.compression_size, self.size_constant], dim=1
)                                               # [B, t+1, 257]
```

`compression_size` counts how many original frames were merged into each bank slot
(starts at 1 per slot; grows when frames are merged).

---

## 7. Visual Memory Bank Compression

When the bank exceeds `memory_bank_length=40` frames, compression runs
(`blip2_vicuna_instruct.py:406-408`, calling `blip2.py::memory_bank_compress` at lines 352-392):

```python
elif self.visual_memory_bank.size(1) > self.memory_bank_length:
    self.visual_memory_bank, self.compression_size = memory_bank_compress(
        self.visual_memory_bank, self.compression_size
    )
```

**How `memory_bank_compress` works** (reduces T slots → T-1 slots):

```
Input:  memory_bank    [B, T, 257, 1408]
        compression_size [B, T, 257]

Step 1 — Find the most similar adjacent pair:
    cosine_similarity(bank[:, :-1], bank[:, 1:], dim=-1)
    → similarity_matrix: [B, T-1, 257]
    
    max over T-1 dim → max_similarity_indices: [B, 1, 257]
    (index of the frame whose right neighbor is most similar)

Step 2 — Weighted-average merge:
    src = bank[max_idx + 1]          # the more similar neighbor
    dst = bank[all other indices]     # everyone else (T-1 slots)

    # Weight by how many original frames each slot represents:
    merged = (src * src_size + dst[at max_idx] * dst_size) / (src_size + dst_size)

Output: compressed_memory_bank [B, T-1, 257, 1408]
        updated compression_size [B, T-1, 257]
```

**Effect over time**: as more frames arrive, the bank keeps the 40 most *informationally
distinct* frame embeddings — redundant/similar frames are silently merged into weighted
averages, not dropped.

---

## 8. Q-Former: Cross-Attention Deep Dive

After accumulating (and possibly compressing) the memory bank, the Q-Former is called
every frame (`blip2_vicuna_instruct.py:398-407`):

```python
mem_hidden = self.visual_memory_bank.view(B, -1, C)
# memory bank:  [B, min(t+1,40), 257, 1408]
# after .view:  [B, min(t+1,40)*257, 1408]   ← encoder_hidden_states

query_output = self.Qformer.bert(
    text_Qformer.input_ids,              # tokenized prompt
    attention_mask=Qformer_atts,
    query_embeds=query_tokens,           # [B, 32, 768]
    encoder_hidden_states=mem_hidden,    # [B, t*257, 1408]
    encoder_attention_mask=mem_atts,
    return_dict=True,
)
```

### Q-Former Architecture

Based on BERT-base:
- 12 Transformer layers
- Hidden size: **768**
- Attention heads: **12** → 768/12 = **64 dims per head**
- Cross-attention inserted every 2 layers (`cross_attention_freq=2`): layers 0, 2, 4, 6, 8, 10
- `encoder_width = 1408` (EVA-CLIP-G feature dim)

### Input Concatenation

At the start, query tokens and text tokens are concatenated into one sequence:

```
hidden_states = [query_tokens  ||  text_tokens]
              = [B, 32, 768]   ||  [B, L_text, 768]
              = [B, 32 + L_text, 768]
```

### Per-Layer Processing (BertLayer)

Each of the 12 layers runs two sub-steps:

#### Sub-step 1: Self-Attention (every layer)

All 32+L_text tokens attend to each other:

```
Q, K, V from hidden_states: [B, 32+L_text, 768]
                              split into 12 heads
                           → [B, 12, 32+L_text, 64]

attention scores: Q @ Kᵀ = [B, 12, 32+L_text, 32+L_text]
output after softmax + V:   [B, 32+L_text, 768]
```

This lets query tokens read the text prompt and vice versa.

#### Sub-step 2: Cross-Attention (layers 0, 2, 4, 6, 8, 10 — query tokens only)

Only the first 32 positions (query tokens) participate in cross-attention.
Text tokens are set aside (`Qformer.py:436`):

```python
query_attention_output = attention_output[:, :query_length, :]  # [B, 32, 768]
```

**Dimension mismatch resolution** (`Qformer.py:134-136`):

The K and V linear layers in cross-attention are defined with `encoder_width` as input:

```python
# Cross-attention K and V projections (input = encoder_width = 1408):
self.key   = nn.Linear(config.encoder_width, self.all_head_size)  # Linear(1408 → 768)
self.value = nn.Linear(config.encoder_width, self.all_head_size)  # Linear(1408 → 768)

# Self-attention K and V projections (input = hidden_size = 768):
self.key   = nn.Linear(config.hidden_size, self.all_head_size)    # Linear(768 → 768)
self.value = nn.Linear(config.hidden_size, self.all_head_size)    # Linear(768 → 768)
```

So the K and V projections from the memory bank are **two sequential steps**:

```
memory bank: [B, t*257, 1408]
      │
      │  nn.Linear(1408 → 768)    ← bridges visual encoder width to Q-Former width
      ▼
             [B, t*257, 768]
      │
      │  transpose_for_scores()   ← reshape into heads: .view(..., 12, 64).permute(0,2,1,3)
      ▼
             [B, 12, t*257, 64]
```

The Q projection (from the 32 query tokens) stays entirely in 768-dim space:

```
query tokens: [B, 32, 768]
      │
      │  nn.Linear(768 → 768)
      │  transpose_for_scores()
      ▼
             [B, 12, 32, 64]
```

**Cross-attention computation** (`Qformer.py:192-270`):

```
Q:  [B, 12,    32, 64]   ← from query tokens
K:  [B, 12, t*257, 64]   ← from memory bank (via Linear(1408→768) then split)
V:  [B, 12, t*257, 64]   ← from memory bank (via Linear(1408→768) then split)

attention scores = Q @ Kᵀ              → [B, 12,    32, t*257]
scores /= sqrt(64)                     → scaled
softmax(scores)                        → attention weights [B, 12, 32, t*257]
                                          each of 32 queries attends over ALL frame tokens

output = weights @ V                   → [B, 12, 32, 64]
concat 12 heads                        → [B, 32, 768]
```

After 6 cross-attention layers, each of the 32 query vectors has selectively
aggregated information from all retained frames in the memory bank.

---

## 9. Query Memory Bank — Per-Layer Self-Attention History

Cross-attention (§8) is only half the story. The paper also defines a **query memory bank** `Z_t`
that lives inside the Q-Former's *self-attention* — one **independent** bank per BERT layer, so
Q-Former with 12 layers has **12 query memory banks** running in parallel with the single visual bank.

### Where it is installed

Not in a subclass file — via a **monkey-patch** at model init time (`blip2.py::apply_memory_bank`,
lines 394-402):

```python
def apply_memory_bank(model, memory_bank_length, num_frames):
    for module in model.modules():
        if isinstance(module, BertSelfAttention):
            if memory_bank_length > 0:
                module.__class__ = MBBertSelfAttention   # ← swap class in place
                module.memory_bank_length = memory_bank_length
                module.num_frames = num_frames
    return model
```

Called from `init_Qformer` (`blip2.py:217`) right after `BertLMHeadModel.from_pretrained("bert-base-uncased", ...)`.
Every one of the 12 `BertSelfAttention` instances gets its class rewritten to
`MBBertSelfAttention` (defined in `blip2.py:46-186`), which is what carries the `self.query_memory_bank` tensor.

### What each layer's bank stores

`MBBertSelfAttention` runs once per frame per layer. On each call it does two things:

**(a) Read history — extend K/V with prior queries** (`blip2.py:72-84`):

```python
k = self.key(hidden_states)                                # [B, 32+L_text, 768]  current step's K
v = self.value(hidden_states)                              # [B, 32+L_text, 768]  current step's V

if hasattr(self, 'query_memory_bank'):                     # true from t=1 onward
    B, T_bank, N_q, C_q = self.query_memory_bank.shape     # [B, T_bank, 32+L_text, 768], T_bank ≤ mbl
    qmb  = self.query_memory_bank.view(B, -1, C_q)         # [B, T_bank*(32+L_text), 768]
    qmb_k = torch.cat([self.key(qmb),   k], dim=1)         # [B, (T_bank+1)*(32+L_text), 768]
    qmb_v = torch.cat([self.value(qmb), v], dim=1)         # [B, (T_bank+1)*(32+L_text), 768]
    key_layer   = self.transpose_for_scores(qmb_k)         # [B, 12, (T_bank+1)*(32+L_text), 64]
    value_layer = self.transpose_for_scores(qmb_v)         # [B, 12, (T_bank+1)*(32+L_text), 64]
```

The current 32 queries then attend over `(T_bank+1)·(32+L_text)` positions — i.e. their own current
tokens plus every previous timestep's query tokens (up to `memory_bank_length`).

**(b) Write history — append the just-processed queries** (`blip2.py:170-184`):

```python
if not hasattr(self, 'query_memory_bank'):
    self.query_memory_bank = hidden_states[:, None, :, :].detach()   # [B, 1, 32+L_text, 768]
    self.size_constant     = torch.ones(B, 1, N_q).to(...)
    self.compression_size  = self.size_constant
else:
    self.query_memory_bank = torch.cat(
        [self.query_memory_bank, hidden_states[:, None, :, :].detach()], dim=1
    )                                                                # [B, T_bank+1, 32+L_text, 768]
    self.compression_size  = torch.cat([self.compression_size, self.size_constant], dim=1)

if self.compression_size.sum(1).mean().round() == self.num_frames:
    del self.query_memory_bank
    del self.compression_size
elif self.query_memory_bank.size(1) > self.memory_bank_length:
    self.query_memory_bank, self.compression_size = memory_bank_compress(
        self.query_memory_bank, self.compression_size
    )
```

### Are compression indices shared between visual and query banks?

**No.** Each of the 13 banks calls `memory_bank_compress` **independently** with its own tensor
and its own `compression_size` counter. Inside the function (`blip2.py:352-392`):

```python
similarity_matrix     = F.cosine_similarity(bank[:, :-1, :], bank[:, 1:, :], dim=-1)  # [B, T-1, N]
max_similarity_indices = similarity_matrix.argmax(dim=1, keepdim=True)                 # [B, 1, N]
```

The `argmax` is computed on **this bank's own contents**. Since:

- The visual bank stores `1408`-dim ViT features across `257` tokens per slot,
- Each query bank stores `768`-dim BERT hidden states across `32+L_text` tokens per slot,
- Each layer's query bank starts from a different self-attention transformation,

the 13 cosine-similarity landscapes look totally different → the `max_similarity_indices` are
different per bank per frame. Not only that: `max_similarity_indices` is shape `[B, 1, N]` — a
**per-token-position** decision — so within a single bank different token positions can even merge
different adjacent pairs.

### Practical consequences

- The single `--memory_bank_length` CLI knob in `infer.py` caps **all 13 banks** at the same size
  (`apply_memory_bank` uses one value for every `MBBertSelfAttention`, and the main model uses the
  same value for the visual bank).
- Peak GPU memory during Q-Former is dominated by the sum of all 13 banks' K/V tensors, not just
  the visual bank. Ablations that raise `memory_bank_length` inflate memory 13× accordingly.
- The `analyze_memory_saturation.py` diversity metric in this repo only inspects the visual bank's
  final state — the query banks are deleted at the last frame (`del self.query_memory_bank`), so
  saturation there is not currently observable without patching.

---

## 10. Projection to LLM Space

After all 12 Q-Former layers (`blip2_vicuna_instruct.py:456`):

```python
query_output.last_hidden_state          # [B, 32 + L_text, 768]

# Slice: keep only the 32 query positions, discard text half
inputs_llm = self.llm_proj(
    query_output.last_hidden_state[:, :32, :]
)
# last_hidden_state slice:  [B, 32, 768]
# llm_proj = nn.Linear(768, 4096)
# inputs_llm:               [B, 32, 4096]
```

The text token slice of `last_hidden_state` is discarded — it was only needed
to condition the query tokens via self-attention during Q-Former processing.

---

## 11. Final LLM Input Assembly

The prompt text is tokenized and embedded through Vicuna's own embedding table
(`blip2_vicuna_instruct.py:468-470`):

```python
inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
# prompt text embedding:  [B, L_prompt, 4096]

# Prepend visual prefix to text:
inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
#                          [B, 32,      4096]
#                                   [B, L_prompt, 4096]
# result:                  [B, 32 + L_prompt, 4096]

attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
# all-ones for visual prefix, real mask for text:
#                            [B, 32 + L_prompt]
```

Vicuna-7B then generates the caption autoregressively from this embedding sequence.
It never sees raw pixel values — only the 32 Q-Former summary vectors prepended to the
text prompt.

---

## 12. End-to-End Tensor Flow Summary

```
infer.py                              blip2_vicuna_instruct.py (generate)
══════════════════════════════════════════════════════════════════════════

Raw frames on disk
  frame000001.jpg ... frame004605.jpg
       │
       │  np.linspace → pick 80 indices, load & transform
       ▼
[T=80, C=3, H=224, W=224]
       │
       │  .permute(1,0,2,3).unsqueeze(0)
       ▼
[B=1, C=3, T=80, H=224, W=224]   ──────────────────────────────────────────────────┐
                                                                                    │
                                  for t in range(80):                               │
                                  ┌─────────────────────────────────────────────┐  │
                                  │                                             │  │
                                  │  image[:, :, t, :, :]                       │  │
                                  │  [B, C, H, W] = [1, 3, 224, 224]            │  │
                                  │        │                                    │  │
                                  │        ▼  EVA-CLIP-G ViT (frozen)           │  │
                                  │  [B, 257, 1408]                             │  │
                                  │        │                                    │  │
                                  │        ▼  ln_vision (LayerNorm)             │  │
                                  │  [B, 257, 1408]                             │  │
                                  │        │                                    │  │
                                  │        ▼  + image_pe(t)  (temporal embed)   │  │
                                  │  [B, 257, 1408]                             │  │
                                  │        │                                    │  │
                                  │        ▼  .unsqueeze(1)                     │  │
                                  │  [B, 1, 257, 1408]                          │  │
                                  │        │                                    │  │
                                  │        ▼  append to memory bank             │  │
                                  │  [B, t+1, 257, 1408]                        │  │
                                  │        │                                    │  │
                                  │        ▼  if bank > 40 frames:              │  │
                                  │     memory_bank_compress()                  │  │
                                  │     merge most-similar adjacent pair        │  │
                                  │  [B, ≤40, 257, 1408]                        │  │
                                  │        │                                    │  │
                                  │        ▼  .view(B, -1, C)                   │  │
                                  │  [B, ≤40*257, 1408]  encoder_hidden_states  │  │
                                  │        │                                    │  │
                                  │   Q-Former (12 BERT layers, per frame)       │  │
                                  │   ├─ self-attention  (all 12 layers) — reads │  │
                                  │   │   & writes this layer's query_memory_bank│  │
                                  │   │   (see §9), current 32+L_text queries    │  │
                                  │   │   attend over (T_bank+1)*(32+L_text) K/V │  │
                                  │   └─ cross-attention (layers 0,2,4,6,8,10)   │  │
                                  │        │                                    │  │
                                  │  Q: query_tokens [B, 32, 768]               │  │
                                  │     Linear(768→768) → 12 heads              │  │
                                  │     → [B, 12, 32, 64]                       │  │
                                  │                                             │  │
                                  │  K: mem_hidden [B, t*257, 1408]             │  │
                                  │     Linear(1408→768) → 12 heads             │  │
                                  │     → [B, 12, t*257, 64]                    │  │
                                  │                                             │  │
                                  │  V: mem_hidden [B, t*257, 1408]             │  │
                                  │     Linear(1408→768) → 12 heads             │  │
                                  │     → [B, 12, t*257, 64]                    │  │
                                  │                                             │  │
                                  │  scores = Q @ Kᵀ                           │  │
                                  │  → [B, 12, 32, t*257]                       │  │
                                  │  softmax → weights @ V                      │  │
                                  │  → [B, 32, 768]    (per frame output)       │  │
                                  │                                             │  │
                                  └─────────────────────────────────────────────┘  │
                                                                                    │
                                  last_hidden_state[:, :32, :]                      │
                                  [B, 32, 768]   (from final frame t=79)            │
                                        │                                           │
                                        ▼  llm_proj = nn.Linear(768 → 4096)         │
                                  inputs_llm [B, 32, 4096]                          │
                                        │                                           │
                                        ▼  torch.cat with prompt text embeddings    │
                                  [B, 32 + L_prompt, 4096]                          │
                                        │                                           │
                                        ▼  Vicuna-7B LLM (frozen)                   │
                                  output tokens → decoded text caption              │
                                                                                    │
══════════════════════════════════════════════════════════════════════════           │
```

---

## 13. Compression Payoff — Numbers

| Path | Visual tokens entering Vicuna |
|---|---|
| **Memory bank ON** (used in `infer.py`) | **32 tokens** — always, regardless of T |
| Memory bank OFF (fallback) | **T × 32 = 80 × 32 = 2,560 tokens** |

Starting from:
- 80 sampled frames
- Each with 257 ViT tokens
- = **20,560 raw visual tokens**

After memory bank + Q-Former: **32 tokens**.

That 640× compression is what makes hour-long Ego4D videos tractable within Vicuna's
context window without truncation.

---

## Key Dimension Reference

| Tensor | Shape | Notes |
|---|---|---|
| Raw frame (one) | `[B, 3, 224, 224]` | After resize + normalize |
| ViT output (one frame) | `[B, 257, 1408]` | 256 patches + 1 CLS, EVA-CLIP-G width |
| **Visual memory bank** | `[B, ≤40, 257, 1408]` | Single, shared; feeds cross-attn |
| Visual bank flattened | `[B, ≤40×257, 1408]` | `encoder_hidden_states` for Q-Former |
| **Query memory bank** (per layer) | `[B, ≤40, 32+L_text, 768]` | **12 independent copies**, one per BERT self-attn layer; feeds self-attn K/V |
| Q-Former query tokens | `[B, 32, 768]` | Learnable, BERT-base width |
| Q-Former K/V (cross-attn) | `[B, 12, t×257, 64]` | After Linear(1408→768) + head split |
| Q-Former Q (cross-attn) | `[B, 12, 32, 64]` | After Linear(768→768) + head split |
| Q-Former self-attn K/V (with query bank) | `[B, 12, (T_bank+1)*(32+L_text), 64]` | Prior queries prepended to current-step K/V |
| Q-Former output (sliced) | `[B, 32, 768]` | Only query positions kept |
| After llm_proj | `[B, 32, 4096]` | Linear(768→4096) |
| Prompt text embeddings | `[B, L_prompt, 4096]` | Vicuna embedding table |
| Final LLM input | `[B, 32 + L_prompt, 4096]` | Prepended visual prefix |
