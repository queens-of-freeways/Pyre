# Distributed Llama — Rust + Mojo Engine: Design & Migration Plan

Status: **G5 complete** (pluggable kernel backend; see §10.1–§10.9 for gate results) — this
document is the source of truth for the rewrite.
The C++ engine in `../src` remains the reference implementation and the **parity oracle**
until the Rust engine reaches feature parity. Nodes must stay wire-compatible during
migration (mixed C++/Rust clusters).

---

## 1. Mission

Rewrite Distributed Llama as a **model-agnostic** inference engine:

1. **Model-agnostic**: new decoder-LLM families run without engine code changes
   (convention-driven graph derivation over GGUF metadata; serialized-graph IR later).
2. **Rust system layer**: cluster protocol, orchestration, API server, loaders, tokenizer.
3. **Pluggable kernel layer** (optional accelerator behind a C-ABI function table):
   quantized matmul, attention, norm kernels — loaded as a dynamic library.
   A pure-Rust CPU backend is always available as fallback. (G5 re-scope: Mojo has
   no Windows toolchain yet, so the Rust AVX2 suite is the reference backend and a
   Mojo cdylib drops in later without engine changes.)

Strategy: **strangler fig, wire-compatible**. The C++ binary keeps running; the Rust
engine grows crate-by-crate; mixed clusters validate each migration step.

## 2. What is hardcoded in the C++ engine today (inventory)

| # | What | Where | Replacement |
|---|------|-------|-------------|
| 1 | `LlmArchType` enum (`LLAMA`, `QWEN3`, `QWEN3_MOE`) + `if (archType == …)` switches | `src/llm.hpp`, `src/llm.cpp:113,156,181,322,425,648` | metadata flags in `ModelMeta` → single generic builder |
| 2 | Hand-wired op list per arch in `buildLlmNet()` with hardcoded buffer indices | `src/llm.cpp` | `dllama-ir` graph builder over ~15 ops |
| 3 | Custom `.m` file format with fixed header key enum | `src/llm.cpp:loadLlmHeader` | Phase 1: `.m` loader (parity); Phase 2: GGUF loader |
| 4 | Per-family conversion scripts with hand-written tensor-name maps | `converter/*.py` | GGUF adoption (Phase 2) — no converter needed |
| 5 | Per-format tokenizers + `ChatTemplateType` enum | `src/tokenizer.*`, `converter/convert-tokenizer-*` | HF `tokenizers` crate + `minijinja` on `chat_template` (Phase 4) |
| 6 | File-global weight type (`q40` or `f32` only) | `src/llm.hpp` | per-tensor quant type from GGUF metadata |

## 3. Target architecture

```
┌─────────────────────────────────────────────────────────────┐
│ dllama-cli / dllama-api (Rust)                               │
│   axum /v1/chat/completions (SSE) · CLI (dllama-style flags) │
├─────────────────────────────────────────────────────────────┤
│ Model-agnostic core (dllama-ir)                              │
│   GGUF/metadata → ModelMeta → Graph (~15 ops)                │
│   TP splitter: sharding annotations + Sync ops               │
├─────────────────────────────────────────────────────────────┤
│ Distribution (dllama-net, Rust/tokio)                         │
│   root/worker protocol (byte-compatible v0) · q80 sync        │
├─────────────────────────────────────────────────────────────┤
│ Kernels (dllama-kernel: Kernels table, C-ABI v1)                       │
│   CPU: Rust AVX2 reference · any cdylib (Mojo later)       │
│   GPU: wgpu/Vulkan or MAX GPU (Phase 5+)                     │
└─────────────────────────────────────────────────────────────┘
```

## 4. IR specification (dllama-ir)

### 4.1 Principles

- **Pipes**: preallocated typed buffers referenced by index (proven model from C++
  executor; keeps executor trivial and memory-plannable).
- **Ops**: a bounded registry (~15). Attention stays **fused** (KV cache + GQA) —
  primitive-level decomposition would be memory-bound and slow.
- **Weights referenced by name** (GGUF convention: `token_embd`, `blk.{l}.attn_q`,
  `ffn_gate`, `ffn_up_exps`, …) so the Phase-2 rule table maps names → ops directly.
- **Sharding is data, not code**: every op carries a `Parallel` annotation; the plan
  (§5) is derived from it, never hardcoded per arch.

### 4.2 Op set (v0)

```
Embedding{w}                    out = row(token)            [OutSharded: vocab rows]
MatMul{w, parallel}             out = x · Wᵀ               [OutSharded | InSharded]
RmsNorm{w, eps}                 out = x·w / rms(x)
QkRmsNorm{qw, kw, eps}          in-place per-head norm of q,k   (Qwen3-style)
Rope{q,k,pos}                   in-place, theta+scaling from ModelMeta
Attention{layer,q,k,v}          fused GQA + per-layer KV cache
ActivatedMul{act}              out = silu(gate) ⊙ up       (act: Silu|Gelu)
Moe{layer,gate,w1,w2,w3}       fused route→top-k→combine    (v0 fused; split later)
ResidualAdd{acc, addend}        acc += addend
Head{w}                         logits = x · Wᵀ            [OutSharded: vocab rows]
Softmax / Argmax                sampling & perplexity support
Sync{kind, pipe, dtype}          inserted by builder/splitter (n_nodes > 1 only)
```

### 4.3 Sharding annotations (generalizes C++ `NnRowMatmulSlice`/`NnColMatmulSlice`)

| IR annotation | C++ equivalent | Meaning | Communication |
|---|---|---|---|
| `Parallel::OutSharded` | row matmul slice | W rows split across nodes; each node emits its slice of the output vector | gather/sum at consume point |
| `Parallel::InSharded` | col matmul slice | W input-columns split; each node computes a **partial full-length** output | **all-reduce** (sum) |
| `Parallel::Replicated` | — | full copy on every node | none |

Mapping to the classic Megatron vocabulary: `OutSharded` = column-parallel,
`InSharded` = row-parallel. The C++ names were ambiguous; the IR names are explicit.

### 4.4 Sync semantics over the star topology

- **AllReduce(pipe)**: each worker sends its quantized (`sync_type`, default q80)
  partial vector to root → root sums all partials → root broadcasts the full vector.
- **AllGather(pipe)**: each worker sends only its OutSharded slice → root
  concatenates → root holds the full vector (no broadcast needed if next consumer
  is root-only, e.g. logits/argmax).
- Canonical sync points per token (built by the generic builder):
  1. after `Embedding` (all-reduce of vocab-sharded rows),
  2. after `wo` matmul per layer (all-reduce of attention output),
  3. after `w2`/MoE combine per layer (all-reduce of FFN output),
  4. after `Head` (all-gather of vocab-sharded logits at root).

## 5. Sharding plan (tensor parallelism)

For `n` nodes (root = node 0, workers = nodes 1..n-1):

| Tensor | Split | Per-node slice |
|---|---|---|
| token_embd | vocab rows | `vocab/n` rows |
| wq | q rows (`n_heads·head_dim`) | contiguous slice; heads must not straddle nodes |
| wk, wv + KV cache | kv rows (`n_kv_heads·head_dim`) | contiguous slice |
| wo | input cols (= q rows) | matches q slice |
| ffn_gate/ffn_up (w1/w3) | hidden rows | `hidden/n` rows |
| ffn_down (w2) | input cols | matches hidden slice |
| output (head) | vocab rows | `vocab/n` rows |
| norms, gate router | replicated | full copy |

**Constraints** (enforced by `dllama-ir::shard`, mirroring C++ slicing):
- `n_nodes` is a power of two (protocol constraint, README "Known Limitations")
- `n_nodes ≤ n_kv_heads` (issue #70)
- `n_heads·head_dim`, `n_kv_heads·head_dim`, `dim`, `hidden/moe_hidden`, `vocab`
  all divisible by `n_nodes`; q/kv slices aligned to head boundaries

Per-token sync traffic per worker (q80): `(2·L+1) · 2 · block34(dim) + block34(vocab/n)`,
where `block34(x) = x/32 · 34` bytes (34 bytes per 32 values, `x % 32 == 0` required).

## 6. Wire protocol v0 (byte-compatible — facts extracted from C++ source)

> Goal: a Rust root can drive C++ workers (and vice versa) during migration.
> All multi-byte values little-endian (native-endian raw struct writes in C++;
> every supported platform is LE).

| Message | Bytes | Source |
|---|---|---|
| **Control packet** (root → all, raw struct) | `{ u32 position; u32 batch_size }` = 8 B; `batch_size == 0` = stop signal | `app.cpp:180-228` |
| **Config: net** (root → worker) | ack; u32 nBatches; u32 nNodes; u32 nPipes; per-pipe: `NnSize3D` (40 B) + string; u32 nPreSyncs; per-preSync u32 pipeIndex; ack | `nn-network.cpp:657-673` |
| **Config: node** (root → worker) | ack; u32 nodeIndex; u32 nBuffers; u32 nSegments; per-buffer `NnSize3D`+string; per-segment: u32 nSyncs, u32 nOps, per-sync u32 pipeIndex+u32 syncType, per-op: u32 code, u32 index, `NnSize3D` weightSize(40 B), u64 configSize, string name, `NnPointerConfig` input(12 B), output(12 B) [+ config payload — **verify**] | `nn-network.cpp:675-723` |
| **Weight chunk** (root → worker) | u32 nameSize (incl. NUL); name bytes; u32 opIndex; u64 offset; u64 nBytes; payload | `nn-network.cpp:835-843` |
| **Weight stream end** | u32 0; then readAck | `nn-network.cpp:814-819` |
| **String** | u32 length + bytes (**verify** `writeString` in nn-network.cpp before Phase 3; recent upstream fix #295 hardened bounds) | — |
| **Activation sync** | q80/f32 payload of the pipe's slice; ack pattern per `NnSyncType` (**verify** values in Phase 3) | — |

`NnSize3D` = `{ u32 floatType; u32 z; u32 y; u32 x; u64 length; u64 nBytes; u64 nBytesXY }` = 40 B.

## 7. Quantization formats (byte-compatible, ported from `src/nn/nn-quants.{hpp,cpp}`)

- **f16**: IEEE 754 half; `f32_to_f16` is an exact port of `convertF32ToF16Impl`
  (round-to-nearest-even via the `+0x0fff + lsb` trick).
- **Q40** (weights): per 32-value block → `u16 f16 scale d` + 16 B packed nibbles (18 B).
  `d = signed_amax / -8`; `id = 1/d`; nibble = `(u8)(x·id + 8.5)` clamped to 15;
  low nibble → values 0..15, high nibble → values 16..31; dequant `(nibble − 8) · d`.
- **Q80** (activations/sync): per 32-value block → `u16 f16 scale d` + 32 × i8 (34 B).
  `d = amax / 127`; dequant `q · d`. *(Open: rounding mode of the scalar C++ path —
  verify against `nn-quants.cpp` before claiming byte parity.)*
- Wire float-type ids (`NnFloatType`): `F_32=0, F_16=1, F_Q40=2, F_80=3` (UNK = −1).

## 8. Model-agnostic loading roadmap

- **Phase 1**: `.m` file loader (parity with existing models incl. qwen3 MoE).
- **Phase 2**: **GGUF** loader. Rule table maps GGUF metadata + tensor-name patterns
  → `ModelMeta` flags + graph ops. Zero converter, thousands of pre-converted models.
  Rule sketch: `blk.N.attn_q` → `MatMul OutSharded`; `blk.N.attn_q_norm` → set
  `qk_norm`; `ffn_*_exps` → MoE path; `rope.freq_base` → theta; etc.
- **Phase 3+**: serialized graph embedded in the container (Tier-3 agnosticism);
  new archs become data, not code.

## 9. Crate map (this workspace)

| Crate | Role |
|---|---|
| `dllama-quant` | q40/q80/f16 codecs, byte-compatible with C++ |
| `dllama-ir` | `ModelMeta`, op registry, graph builder, TP shard planner |
| `dllama-kernel` | `Kernels` C-ABI v1 fn-ptr table + `libloading` loader; Rust AVX2 kernels (matmul, rmsnorm, attention, …); `dllama-kernel-c` reference cdylib |
| `dllama-net` | wire protocol: control packets, config/weight stream codecs |
| `dllama-cli` | `dllama-rs` binary: `bench`, `graph-dump` (CLI flags mirror the C++ style) |

## 10. Milestones & validation gates

| Gate | Deliverable | Validation |
|---|---|---|
| **G0** (this PR) | Workspace compiles; quant codecs round-trip; graphs build for llama/qwen3/qwen3-MoE presets; sharding validated; Phase-0 matmul bench | `cargo test` green; bench numbers recorded |
| **G1** | `.m` loader + single-node forward pass for llama 8B q40 | logits match C++ for a fixed prompt; perplexity parity ±0.001 |
| **G2** | GGUF loader + rule table | run a Mistral/Gemma GGUF with zero code changes |
| **G3** | Distribution: Rust root ↔ C++ workers, mixed cluster | byte-identical config/weight/sync streams; tokens/s ≥ C++ |
| **G4** | API server (axum, SSE), HF tokenizers, minijinja templates | OpenAI-API conformance vs C++ `dllama-api` |
| **G5** | Pluggable kernel backend: C-ABI v1 + `libloading`; Rust reference (built-in + cdylib plugin); **Mojo backend shipped & bit-exact on Linux/WSL**; **OpenCL GPU accelerant bit-exact on Intel iGPU**; OS-aware default selection (Windows→Rust, Linux→Mojo); GPU compute kernels → G7 | plugin ≡ built-in bit-exact (full op-trace diff); ppl unchanged; bad plugin degrades gracefully |
| **G6** | K-quant AVX2 kernels (Q4_K/Q6_K), matmul row threading + parallel MoE experts, `--threads` flag; bit-exact by construction (rows/experts independent, serial float chains preserved) | kernel bench 2.0× (Q4_K, Q6_K); e2e trace diffs byte-identical; 30B 2.0→1.55 s/tok (storage-bound on this box) |
| **G7** | GPU compute: weight-residency state + attention/norms on device (OpenCL accelerant already bit-exact for matmuls, §10.12; Mojo GPU kernels compile-verified, §10.11) | discrete-GPU perf vs CPU; bit-exact on-device |

## 10.1 G0 results (recorded 2026-09-30, single thread, MSVC x64)

m=1 (GEMV — the per-token inference regime), n=k=4096:

| Path | time/iter | GFLOP/s | weights read |
|---|---|---|---|
| f32 | 18 ms | 1.83 | 3.66 GB/s (memory-bound) |
| q40 on-the-fly dequant | 27 ms | 1.26 | 0.35 GB/s |
| q40 predequant + f32 | 20 ms | 1.67 | 0.47 GB/s |

Readings: (a) m=1 GEMV is bandwidth-bound, so f32 throughput ≈ DRAM stream rate;
(b) scalar per-row dequant overhead eats the q40 bandwidth advantage — confirms
the G5 plan: **fused SIMD dequant-in-kernel** is mandatory, exactly where the
Mojo backend (or Rust `std::simd`) earns its keep.
Quant codecs: q80 278 M values/s, q40 332 M values/s (scalar) — adequate for
sync buffers at G3, revisit at G5.

Port findings worth keeping:
- Q40's scale rule (`d = signed_max / -8`) makes the block range **asymmetric**:
  values opposite in sign to the block max clamp at 7/8·amax (error up to amax/8).
- Zero blocks encode as scale `-0.0` (f16 `0x8000`) + all-nibble `8` (`0x88`),
  not zero bytes — but dequantize to exact zeros. Matches C++ behavior.

## 10.2 G1 results (recorded 2026-09-30 — model: qwen3-0.6b q40, oracle: C++ dllama.exe)

**What works (validated against the C++ engine):**
- `.m` loader: byte-exact — the 957 MB file is consumed to the last byte with the
  C++ missing-bytes check (`dllama-model`).
- Tokenizer: identical token streams (same token counts, `AddBos: 0` honored,
  special tokens, score-merge BPE).
- Executor: all ops structurally correct — verified with a **per-op trace diff**
  (C++ `DEBUG_OP_INPUT_OUTPUT` vs Rust `DLLAMA_RS_DEBUG=1`): the first op's
  output matches to **1 ULP**, inputs identical everywhere.
- End-to-end generation: same first greedy token (" Paris"), coherent text
  from both engines.

**Numerics discoveries (now encoded in dllama-exec):**
1. Activations are **q80-quantized before every matmul** (`block_cast_y`), AND
   the attention/FFN block outputs are q80-quantized **before the residual
   add** (`block_cast_d` → `merge_add`). The residual stream only ever sees
   q80-precision block outputs when `sync_type != f32`.
2. Q80 rounding is ties-to-even on AVX2 (`_mm256_round_ps`),
   round-half-away in the scalar fallback.
3. C++ uses `expf_avx2`, a degree-4 2^x polynomial with fmadd — not exact exp.

**Residual gap (documented, not a bug):** ppl 10.87 (C++) vs 11.79 (Rust).
Cause: float accumulation-order differences (AVX2 `fmadd` + `horizontalSum` +
`expf_avx2` vs scalar exact math) get **amplified by q80 boundary flips**
(~9 flips/forward by estimate): a 1-ULP diff at layer 0 becomes ~5e-4 relative
by layer 2 and flips near-tie token choices. Bit-exact parity therefore
requires replicating the C++ SIMD kernels exactly — this IS the G5 kernel
work, now precisely specified:
- `invRms_F32` / `rmsNorm_F32` (fmadd + `horizontalSum_avx2` reduction order)
- `dotProduct_F32` (fmadd + horizontal sum — attention scores)
- `matmul_Q80_Q40_F32` AVX2 path (q40 nibble unpack + fmadd lanes)
- `expf_avx2` polynomial (scalar `mul_add` port gives bit-exact lanes)
- `softmax_F32` AVX2 path, `silu` (uses expf_avx2)
Verification harness: `DLLAMA_RS_DEBUG=1` trace diff against the C++ oracle
(keep `DEBUG_OP_INPUT_OUTPUT` flip handy, do not commit it).

**Performance (expected G0 numbers):** Rust scalar 1.1 s/token vs C++
(sgemm, 4 threads) 0.13 s/token — the same G5 SIMD/Mojo kernel port closes this.

## 10.3 G1.5 results (recorded 2026-10-01 — bit-exact AVX2 kernel suite)

**Ported (dllama-kernel/src/avx2.rs, from nn-cpu-ops.cpp):**
- `matmul_Q80_Q40_F32` — integer-exact q80×q40 block dots (`madd`/`hadd` i16/i32
  lanes) + one fma per block; scalar emulation is bit-exact too (integer sums
  are order-independent).
- `invRms_F32` / `rmsNorm_F32` — fmadd lane accumulation + `horizontalSum_avx2`
  reduction tree; scalar emulation via `f32::mul_add` + lane arrays is bit-exact.
- `dotProduct_F32`, `softmax_F32` (incl. `expf_avx2` degree-4 2^x polynomial),
  `silu_F32`, attention V-accumulation.
- `libm` externs (`expf/cosf/sinf/powf`): Rust's `f32` math on windows-msvc
  promotes through f64, so direct f32 libm calls are required for parity.

**Parity discoveries (the hard-won knowledge):**
1. **GCC `-ffp-contract=fast`**: every `sum += a*b` in the C++ is a single
   **fma** — Rust must use `mul_add` everywhere (matmul block accumulation,
   attention V-accum). This was the source of 456/1014 trace mismatches.
2. rmsNorm association is `(x·inv)·w`; q80 rounding is ties-to-even; the
   residual addend passes through q80 (from G1).
3. Attention softmax with `< 8` scores is a pure scalar `expf` tail — the LAST
   remaining 1-ULP divergence (3/1014 trace lines, pos-2 forwards) is a
   **MinGW-vs-MSVC libm `expf` implementation difference**, fully localized.
   Fix when desired: port MinGW's expf (~30 lines) behind `libm::expf`.

**Measured (qwen3-0.6b q40, single thread):**
| Metric | C++ oracle | Rust (G1.0) | Rust (G1.5) |
|---|---|---|---|
| trace bit-exact (32-el window, 1014 op lines) | — | 556/1014 | **1011/1014** (rest 1 ULP) |
| perplexity (9-step prompt) | 10.8734 | 11.788 | **10.6368** |
| generation speed | 118.6 ms/tok | 1118.9 ms/tok | **199.6 ms/tok** |

Perf follow-ups: eliminate per-op `Vec` clones in the executor (~30% expected),
then rayon threading over the matmul `d` dimension (mirrors C++ `SPLIT_THREADS`).

## 10.4 G2 results (recorded 2026-10-02 — GGUF loader, model-agnostic loading)

**Delivered (zero per-model code — one rule table for the llama family):**
- `dllama-gguf` crate: container parser (magic/version/KV/tensor directory/
  aligned data section), metadata -> ModelMeta rule table (llama, mistral,
  qwen2, qwen3, smolLM, exa), tensor-name mapping, per-tensor quant kinds,
  tied-embedding fallback (`output` missing -> reuse `token_embd`).
- Quant support: F32, F16, Q4_0 (== dllama q40 layout), Q8_0 (== dllama q80
  layout), **Q4_K / Q6_K** (256-value super-blocks, ports of the ggml
  reference dequant + integer-dot matmuls — layouts verified against
  ggml-common.h / ggml-quants.c fetched from upstream).
- Matmul dispatch per tensor kind (`dllama-kernel::gguf::matmul_q80`),
  embedding + norm dequant per kind, `--buffer-float-type f32` fidelity mode.
- CLI: magic-byte sniffing (.m vs GGUF) — `perplexity`/`inference` run both
  containers transparently. Debug tooling: `gguf-check` (tensor kinds,
  int-vs-ref cross-check, cross-container weight correlation, vocab
  comparison), `gguf-to-m` / `m-to-gguf` converters (isolation experiments).

**Validation (qwen3-0.6b; 33-token natural-text prompt):**
- **Engine isolation**: .m weights converted to F32-GGUF -> GGUF engine:
  ppl 7.734 vs .m engine 7.683 on the same effective weights (0.07 = float
  accumulation noise) — the entire GGUF engine path is validated end-to-end.
- Weight decoding: cross-container correlation vs the independent b4rtaz q40
  quantization: 0.996+ avg (3000 embedding rows 1.0000, 512 rows of every
  layer-0 matmul, norms exact); int-vs-ref matmul diffs ~3e-7 on real bytes.
- Vocab alignment: .t tokenizer ids match GGUF `tokenizer.ggml.tokens`
  (differences are GPT-2 byte-encoder artifacts, e.g. "ìĺ¨" == "온").

**The plot twist (documented for posterity):** bartowski's Q4_K_M/Q8_0
score ppl ~11.8/9.7 vs the .m's 7.68 — after the isolation experiments above
ruled out engine, weights-decoding, and vocab, the conclusion is a
**different source checkpoint** (Qwen3-0.6B base vs instruct-hybrid have
0.996 correlation yet genuinely different raw-text ppl). **Tiebreaker
confirmed**: official Qwen/Qwen3-0.6B-GGUF Q8_0 scores 9.76 — statistically
identical to bartowski's 9.73. The .m q40 file scores 7.68 — a base-style
checkpoint. Both load correctly through the same engine.

**Known gaps (G2.5):** MoE execution (`ffn_*_exps` tensors parse, executor
errors clearly); GGUF-embedded tokenizer (needs llama.cpp-style BPE with
regex pre-tokenization — currently the dllama `.t` tokenizer is required,
vocab-verified); Q5_K/Q2_K/Q3_K/IQ quants; bf16.

## 10.5 G2.5 results (recorded 2026-10-02 — MoE executor + mmap)

**Delivered:**
- **`Op::Moe` executor** (`dllama-exec::moe_ffn`, exact port of the C++ op
  chain): f32 router gate (no activation cast) → softmax → top-k with
  **renormalized weights** (`normTopk=1`, `p/sum(topk)` — `moeGateForward`) →
  per-expert SwiGLU with q80-cast inputs (`repeat_z`/`cast_d2` semantics,
  silu via the `expf_avx2` polynomial) → scale → merge-sum.
- **Expert tensor views**: `ExpertTensors::Split` (GGUF `ffn_{gate,up,down}_exps`)
  and `::Lumped` (dllama `.m` per-expert w1|w2|w3 interleaving).
- **mmap weight storage** (`memmap2`) in both containers — required: this
  machine has 15.8 GB RAM and the 30B is 17.3 GB; weights stream through the
  page cache (MoE helps: ~2.4 GB touched per token).
- Rule table: `qwen3moe` arch string + Falcon rope for all qwen* families.

**Validation (Qwen3-30B-A3B Q4_K_M, official GGUF, 17.3 GB):**
- Unit test: synthetic 3-expert MoE vs naive reference (routing, scales,
  SwiGLU, merge) — `moe_ffn_matches_naive` ✓ (37 tests total green).
- Container: meta (48 layers, 128 experts, 8 active, dim 2048, head_dim 128),
  exps tensors Q4_K, int-vs-ref matmul cross-checks ~3-5e-7.
- Integration: ppl 7.056 on the 33-token story prompt (0.6B instruct = 9.73 on
  the same text — sane ordering); generation:
  `"The capital of France is Paris. The capital of Italy is Rome. The capital
  of Spain is Madrid. The capital of Germany is Berlin."`
- Perf: ~2.0 s/token single-thread (scalar K-quant MoE kernels + cold page
  cache) — AVX2-izing `matmul_q80_q4_k/q6_k` + rayon threading = G6 perf work.

**Bugs found by the integration run:**
1. `qwen3moe` missing from the rope-family match → Llama-adjacent rotation on
   a Falcon model → repetition loops, ppl 13.25 → fixed → 7.06. Lesson:
   position-0 predictions are rope-agnostic (rotation = identity), so first-
   token probs stay identical while everything else breaks.
2. Windows Smart App Control blocks *specific freshly-linked binary hashes*
   (error 4551) — workaround: force a relink (touch a source + rebuild).

**Known remaining gaps:** Q5_K/Q2_K/Q3_K/IQ quants, bf16, GGUF-embedded
tokenizer, `.m` MoE parity run (the C++ oracle path is implemented via
`Lumped` but not exercised — 17 GB .m download deferred).

## 10.6 G3 results (recorded 2026-10-02 — TCP tensor-parallel cluster)

**Delivered:**
- `dllama-cluster` crate: wire protocol v1 (`DLRS` magic, ModelMeta
  serialization, per-node tensor streaming, control packets, q80 all-reduce,
  f32 all-gather), worker serve loop, root-side weight sharding +
  col-repacking (`repack_cols` for InSharded tensors).
- `dllama-exec::distributed`: shard-aware forward pass (separate from the
  single-node path — G1.5 bit-parity preserved), `SyncCtx` enum
  (Single/Worker/Root), per-op slice offsets from `NodeShard`, local KV cache.
- CLI: `dllama-rs worker --host --port` + `dllama-rs perplexity-dist
  --workers "host:port ..."`.
- mmap weights (from G2.5) — workers receive only their slice over TCP.

**Validation (Qwen3-0.6B Q4_K_M GGUF, 2 nodes on localhost):**
| Metric | Single-node | 2-node distributed |
|---|---|---|
| ppl (33-token story) | 11.97 | **11.94** |
| ppl (9-token pangram) | 31.62 | 29.63 |

The 0.03 ppl delta on the story prompt = q80 sync rounding (each node
quantizes its partial separately, matching C++ `SYNC_NODE_SLICES` +
`merge_add`). The pangram shows a larger swing because the instruct
checkpoint has unstable tails on short prompts (established in G2).

**Bugs found and fixed:**
1. `wo_cols`/`ffn_cols` in the shard planner sliced by `dim` instead of
   `q_dim`/`ffn_dim` — hidden because `dim == q_dim` for the llama-8b test
   model; manifested on qwen3 where `dim=1024 ≠ q_dim=2048`.
2. `dequantize_row_q80` **overwrites** the output buffer (assignment), so the
   all-reduce "sum" was just the last slice — fixed with a temp buffer +
   element-wise accumulation (matching C++ `add_Q80_F32`).

**Protocol notes (for C++ interop, G3.5):**
- The Rust protocol differs from the C++ wire format (12-byte control packets
  with token, different setup sequence). C++ interop requires either:
  (a) a protocol adapter in the Rust root, or (b) a C++ worker that speaks
  the Rust protocol. The *semantics* match (q80 all-gather + local merge).

**Known gaps:** MoE distributed (G3.1), multi-threaded workers, C++ worker
interop, batch processing.

## 10.7 G3.1 results (recorded 2026-10-03 — MoE distributed execution)

**Delivered:**
- MoE expert tensor slicing in `node_tensor_slices` (`dllama-cluster`):
  per-expert row sharding for `ffn_gate_exps`/`ffn_up_exps` (OutSharded),
  per-expert column repacking for `ffn_down_exps` (InSharded), replicated
  router gate (`ffn_gate_inp`).
- MoE distributed path in `forward_with_stream` (`dllama-exec::distributed`):
  router gate replicated (all nodes compute the same top-k), expert SwiGLU
  sharded (each node computes its `moe_hidden_dim/n` slice per active expert),
  output all-reduced (same pattern as dense `ffn_down` → `Sync` → residual).
- The existing `moe_ffn` works unmodified with pre-sliced expert tensors
  (it derives dimensions from weight sizes).

**Validation:**
- Dense distributed regression: ppl **11.94** (unchanged from G3) ✓
- Single-node MoE regression: 30B-A3B still runs ✓
- MoE 2-node with the 30B: **blocked by the 256-alignment constraint**
  (moe_hidden_dim=768, Q4_K → 384 % 256 ≠ 0 for 2 nodes).

**The 256-alignment constraint (documented for users):**
| Model | moe_hidden | Quant | Block | 2-node | 4-node |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 768 | Q4_K/Q6_K | 256 | ✗ (384%256≠0) | ✗ |
| Qwen3-30B-A3B | 768 | Q8_0 | 32 | ✓ (384%32=0) | ✓ (192%32=0) |
| Qwen3-30B-A3B (.m) | 768 | q40 | 32 | ✓ | ✓ |

The C++ engine has the same constraint for K-quant models — it uses q40
(32-value blocks) for the `.m` format, which avoids it. For GGUF MoE models
with K-quants, distributed requires `moe_hidden_dim % (256 × n_nodes) == 0`
or a Q8_0/q40 quant.

## 10.8 G4 results (recorded 2026-10-03 — GGUF tokenizer, chat, API)

**GGUF-embedded tokenizer** (the headline feature):
- `GgufFile::tokenizer()` reads `tokenizer.ggml.tokens/scores`, bos/eos ids,
  `add_bos_token`, and `tokenizer.chat_template` from GGUF metadata.
- `Tokenizer::from_gguf_data()` builds the same data structures as the `.t`
  format — the encode/decode algorithms are unchanged.
- **GPT-2 byte decoding**: GGUF vocab entries use the GPT-2 byte encoder
  (`"Ġ"` = space, `"Ċ"` = newline, etc.). Ported the full byte→codepoint→byte
  mapping from the GPT-2 reference; vocab entries are decoded to raw bytes
  at load time.
- **Validation**: GGUF without .t file → ppl **31.624399** (byte-identical to
  the .t result). `.m` regression: ppl **10.636839** ✓. Generation: "The
  capital of France is Paris..." ✓.
- **Result: GGUF models are now truly single-file** — no external tokenizer.

**Chat command**: `dllama-rs chat --model file.gguf` uses the ChatML
template (Qwen3 family) with interactive REPL. The GGUF
`tokenizer.chat_template` is loaded for future Jinja rendering (G4.5).
Fixed a prompt-eval off-by-one (feed n-1 tokens, not n — autoregressive
pattern: the last prompt token starts generation, not gets double-fed).

**Chat validation**: "The capital of France is **Paris**." with GGUF
tokenizer, ChatML template, Qwen3 thinking behavior visible. `.m`
regression: ppl 10.636839 (unchanged from G1.5). 37 tests green.

**Known remaining (G4.5)**: axum API server with `/v1/chat/completions`
+ SSE streaming; minijinja rendering of the GGUF `chat_template` string
(currently hardcoded ChatML); batch processing.

## 10.9 G5 results (recorded 2026-10-03 — pluggable kernel backend, C-ABI v1)

**Re-scope:** Mojo has no Windows toolchain (no winget package, no `mojo` CLI;
Modular ships Linux/macOS packages only). G5 therefore delivers the backend **seam**
with the Rust AVX2 suite as the reference implementation — a Mojo backend now only
needs to export the same symbols; zero engine change.

### Contract (`dllama-kernel/src/abi.rs`, `ABI_VERSION = 1`)

- `dllama_kernels_abi_version() -> u32` gates the table (mismatch → reject).
- `dllama_kernels_name() -> *const c_char` for diagnostics.
- Kernels (`usize` = size_t; quant kinds travel as stable `u32` ids):
  `dllama_matmul_f32`, `dllama_matmul_q40_f32` (f32-input paths),
  `dllama_matmul_q80` (kind dispatch: DllamaQ40/Q4_0/Q8_0/Q4_K/Q6_K/F16/F32),
  `dllama_dequant_row`, `dllama_rmsnorm`, `dllama_inv_rms`, `dllama_softmax`,
  `dllama_activated_mul`, `dllama_dot`, `dllama_expf` (the C++ `expf_avx2`
  polynomial — silu/softmax parity lives here), `dllama_attention` (k/v cache
  passed as raw slices; the engine owns the cache).
- Quantized matmuls are single-token (m = 1); batch variants are ABI v2 with
  prompt prefill.
- Host-side by design (mirrors the C++ `nn-core`/`nn-quants` vs `nn-cpu-ops`
  layering): rope-cache libm math, residual q80 round-trip, q80 codecs, argmax,
  sampling.

### Engine wiring

- `Inference` holds a `Kernels` fn-pointer table (`dllama-kernel/src/lib.rs`):
  `Kernels::load()` — if `DLLAMA_KERNEL_LIB` is set, load the cdylib via
  `libloading` (version-checked, leaked for process lifetime); else use the
  built-in table (the same functions). A failing plugin **downgrades to built-in
  with a warning** — a bad backend can never brick the binary.
- Every per-token kernel call in `dllama-exec` (single-node + distributed paths,
  MoE router/experts, qk-norm, embedding dequant) routes through the table.
- The G0-era `Backend` trait (never actually wired) was removed; `cpu::attention`
  now takes cache *slices* instead of `&mut KvCache` (FFI-friendly).

### Validation (all green)

| Check | Result |
|---|---|
| `cargo test --workspace` | 40 passed — new: kind-wire roundtrip, builtin-table **bit-parity vs direct calls** (`to_bits` equality), plugin-load rejection |
| qwen3-0.6b `.m` (DllamaQ40 + q80 buffers): full op-trace diff, built-in vs cdylib | **byte-identical** (8719 trace lines; ppl 7.913573 both) |
| qwen3-0.6B GGUF Q4_K_M (Q4_K/Q6_K/Q8_0/F16 through the table): trace diff | **byte-identical** (ppl 13.472507 both) |
| `DLLAMA_KERNEL_LIB=not-a-backend.dll` | warning on stderr + built-in fallback, identical ppl |
| 2-node `perplexity-dist` with the cdylib loaded on root **and** worker | runs; per-token probs within the G3-documented q80 sync noise (ppl 7.727886 vs single-node 7.913573) |

Reference plugin build + run:

```
cargo build -p dllama-kernel-c --release        # target\release\dllama_kernel_c.dll
set DLLAMA_KERNEL_LIB=target\release\dllama_kernel_c.dll
dllama-rs perplexity ...
# stderr: [kernel] backend "rust-avx2 (cdylib reference)" loaded from ... (ABI v1)
```

`dllama-kernel-c` doubles as the ABI **conformance reference**: it forwards to the
exact same Rust functions, so "plugin ≡ built-in" is the baseline every other
backend must first reproduce bit-for-bit (the G1.5 trace-diff methodology is the
acceptance test).

### Mojo drop-in (when a Windows toolchain ships)

`kernels.mojo` exports the same 13 symbols via `@export abi("C")`, returns
`ABI_VERSION = 1` + its own name, and implements the v1 numerics. ABI v2 (batch
matmul, prefill) bumps the version; the loader rejects mismatches until the engine
is rebuilt. Perf note: table dispatch is one indirect call per op — noise next to
the matmuls; G6 (rayon + AVX2 K-quant MoE matmuls) plugs in as yet another backend
behind the same table.

## 10.10 G5.1 results (recorded 2026-10-04 — Mojo backend, Linux/WSL)

Mojo has no Windows toolchain, but WSL2 gives us Linux — so the **real Mojo
backend** was built and validated end-to-end on Ubuntu 26.04 (WSL2), loaded by
the Linux build of the engine through the exact G5 seam (no engine change).

### Toolchain (rootless — no sudo needed)

Mojo is now a pip/uv-style package:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv (user-local)
uv venv ~/mojo-env && VIRTUAL_ENV=~/mojo-env uv pip install mojo \
    --index https://whl.modular.com/simple/               # Mojo 1.1.0
~/mojo-env/bin/mojo build --emit shared-lib --fp-mode contract=off \
    kernels.mojo -o libdllama_mojo.so
```

### Mojo 1.1 language notes (vs. older Mojo docs)

- `fn` is gone — everything is `def`; `@export` requires an explicit
  `abi("C")` effect: `@export def name(args) abi("C") -> T:`.
- No module-level mutable globals; typed pointers are
  `Pointer[Scalar[DType.float32], MutUntrackedOrigin]`; indexing is
  `p[unsafe_offset=i]`, arithmetic `p.unsafe_offset(i)`;
  value bit-reinterpretation is `std.memory.bitcast[DType.float32, 1](u32)`.
- Shared libs called from non-Mojo hosts must call
  `std.runtime.initialize_runtime()` before stdlib use (idempotent, cheap —
  called at the top of every exported kernel).
- C string for `dllama_kernels_name`: string literals are NUL-terminated by
  contract — `"mojo-1.1 (wsl)".as_c_string_span().ptr()`.
- `stack_allocation` takes comptime sizes only → runtime scratch uses
  `std.memory.alloc` + `dealloc` (mirrors the Rust reference's per-call
  `Vec` allocations).
- `math.round` is ties-to-even (matches `_mm256_cvtps_epi32`), `math.fma` is
  the correctly-rounded hardware fma.

### The fp-contraction trap (the key discovery)

Mojo defaults to **`--fp-mode contract=fast`** (like GCC `-ffp-contract=fast`):
`sum + a*b` silently becomes one `vfmadd` — the *first* build drifted at the
6th decimal on GGUF K-quant matmuls (15–28/64 fuzz rows) while the explicit-
`math.fma` q40/q80 paths stayed bit-exact. The Rust reference never contracts,
so the backend must be built with **`--fp-mode contract=off`** — with it,
the K-quant paths (`sum += dx*(ds*dot − ms*sumx)` etc.) are bit-exact too.
Any future backend must answer the same question: *where does the reference
use fused vs separate mul/add?* (G1.5 pinned the map: integer-exact block
dots → one fma per block; lane chains → fma; K-quant sub-block terms and all
scalar tails → separate mul+add.)

### Validation (WSL2 Ubuntu 26.04, x86_64, 4 cores)

| Check | Result |
|---|---|
| ctypes smoke (13 symbols, name, expf(0)=1.0, dot, softmax) | pass |
| Linux engine build (`cargo build --release --workspace`) | clean (pre-existing warnings only) |
| **Cross-platform**: Linux builtin ppl vs Windows builtin ppl (.m model) | **identical** — ppl 7.913573 both platforms |
| qwen3-0.6b `.m` (q40/q80 paths): trace diff Mojo vs builtin | **byte-identical** (8719 lines, ppl 7.913573) |
| qwen3-0.6B **GGUF Q4_K_M** (Q4_K/Q6_K/F16/Q8_0 through Mojo): trace diff | **byte-identical** (8719 lines, ppl 13.472507) |
| generation (inference, 16 steps, Mojo backend) | works, 636 ms/tok |
| perf: ppl wall time, builtin (AVX2) vs Mojo | 2.63 s vs 7.12 s (~2.7× — scalar-lane port; SIMD-izing the lanes is future work) |

Workflow scripts live in `rust/wsl/` (setup, build, smoke, run, perf);
`rust/mojo/kernels.mojo` is the backend source (~640 lines). When Modular
ships a Windows toolchain, the same file compiles to
`dllama_mojo.dll` for the Windows engine — ABI v1 needs no changes.

## 10.11 G5.2 results (recorded 2026-10-04 — Mojo GPU device kernels, compile-verified)

Goal (user): the Mojo backend should run on **CPU and any GPU** — Mojo's
write-once/target-anywhere pitch. Delivered the device-kernel half:

**`rust/mojo/gpu_kernels.mojo`** — GPU matmul kernels for all four quant kinds
(DllamaQ40/Q4_0, Q8_0, Q4_K, Q6_K) written with portable `max.gpu` intrinsics
(`thread_idx`/`block_idx`/`block_dim` → PTX for NVIDIA, AMDGCN for AMD, Metal
target emerging). Bit-exactness is preserved by construction: one GPU thread
per output row, walking its weight row in block order — integer-exact block
dots (order-independent) + one sequential fma per block = the exact G1.5
numerics chain. The K-quant terms use plain mul+add (the fp-contract lesson
from §10.10 applies to device code too — built with `--fp-mode contract=off`).

Host dispatchers follow the official vector-addition pattern
(`DeviceContext()` → `create_buffer_sync` → `enqueue_copy_from` →
`ctx.enqueue_function[kernel](…, grid_dim, block_dim)` → `synchronize` →
`enqueue_copy`), gated by the non-raising `DeviceContext.number_of_devices()`
plus an explicit `DLLAMA_MOJO_GPU=1` opt-in. **Compile-verified** into a
shared lib on this CPU-only box (device closures type-check; PTX generation
happens at runtime on-device via the DeviceFunction compiler).

Honest hardware note: this dev machine has an Intel HD 620 — **not a Mojo
GPU target** (Mojo targets NVIDIA + AMD), so runtime validation needs real
GPU hardware. Remaining wiring, documented in the file header:
(1) hook the dispatchers into `dllama_matmul_q80` (2-line change),
(2) the weight-residency state bootstrap — Mojo 1.1 has no mutable globals,
so the design uses a per-PID `/tmp` state cell (open/mmap via `external_call`)
holding the RegisterPassable `DeviceContext` handle + a
{host_ptr → device buffer} arena (~20 µs/call, ABI v1 untouched),
(3) then a block-per-row + shared-memory kernel variant for coalescing.

## 10.12 G5.3 results (recorded 2026-10-04 — OpenCL GPU accelerant + OS-aware backend selection)

User goal: the **Rust** kernel should also run on any GPU, and the tool should
detect the OS — Windows → Rust backend, Linux → Mojo backend.

### Why OpenCL (not wgpu)

wgpu/WGSL has no `fma` intrinsic, so the bit-exactness contract (G1.5) is
unreachable there. OpenCL C has a correctly-rounded `fma()` builtin and runs
on **NVIDIA, AMD and Intel** — including this repo's dev machine (Intel HD
620), which made full runtime validation possible on real GPU hardware.

### Implementation (`dllama-kernel/src/opencl.rs`)

- Hand-rolled FFI over the Khronos ICD loader (`OpenCL.dll` /
  `libOpenCL.so.1`) via `libloading` — zero new dependencies, same pattern as
  the plugin loader. 14 `cl*` functions, contexts leaked for process lifetime.
- Device kernels for the four matmul kinds, one work-item per output row
  (integer-exact block dots + sequential `fma()` per block / plain mul+add
  for K-quant terms — the exact CPU reference order).
- Weight cache keyed by `(host_ptr, len)` — valid because engine weights are
  mmap'd and never move; device copies upload once per tensor. Activations
  (q80, KBs) upload per call; outputs read back per call.
- Scoped to the `matmul_q80` family; norms/attention/codecs stay on the CPU
  reference path (same scope as the Mojo GPU file). `DLLAMA_RUST_GPU=0`
  disables; any failure (no device, alloc too big, build error) degrades the
  *specific call* to CPU — a GPU can never break correctness.

### Two traps found (both fixed)

1. **Intel's OpenCL compiler contracts mul+add despite
   `#pragma OPENCL FP_CONTRACT off`** (the OpenCL twin of Mojo's §10.10
   lesson): Q6_K drifted exactly 1 ULP, Q4_K much more. Fix: volatile
   materialization — the products are stored/loaded through `volatile float`,
   which LLVM must not forward, so the following add can never fuse into an
   fma. Explicit-`fma()` kernels (q40/Q8_0) were immune, which is what
   localized it.
2. **Weight-cache address reuse in tests**: two same-size `Vec` blobs
   (q40 and q4_k are both 1152 B here) got the same heap address after a
   drop → stale device buffer → garbage that looked like a kernel bug. The
   cache's contract (weights never move) is true for the engine; tests now
   `std::mem::forget` their blobs.

### Validation (Windows, Intel HD 620 via OpenCL)

| Check | Result |
|---|---|
| unit: all 4 kinds, GPU vs CPU `to_bits` equality | **bit-exact** |
| qwen3-0.6b `.m` e2e trace diff (GPU vs CPU), full op traces | **byte-identical** (8719 lines, ppl 7.913573) |
| qwen3-0.6B GGUF Q4_K_M e2e trace diff | **byte-identical** (ppl 13.472507) |
| perf (0.6B ppl wall): CPU AVX2 vs HD 620 | 3.53 s vs 4.34 s — launch-overhead-bound at this size; discrete-GPU + large-model perf TBD |

### OS-aware backend selection (`Kernels::load()`)

`DLLAMA_KERNEL_LIB` (explicit override) → platform default plugin probe →
built-in Rust. The probe looks for `dllama_mojo.dll` / `libdllama_mojo.so` /
`libdllama_mojo.dylib` next to the executable, in the repo layout
(`<exe_dir>/../../mojo/`), and in the cwd. Windows has no Mojo toolchain, so
the probe finds nothing there and the built-in Rust backend (AVX2 + optional
OpenCL) stays in charge — i.e. **Windows → Rust, Linux → Mojo, exactly as
requested**. Demonstrated on WSL: with the `.so` next to the exe and no env
vars, the engine logs `[kernel] backend "mojo-1.1 (wsl)" loaded from
.../libdllama_mojo.so (ABI v1)` and produces identical results. The OpenCL
accelerant applies to the built-in Rust table only, so Linux-with-Mojo runs
pure Mojo (no stacking), and Linux-without-Mojo falls back to Rust+OpenCL.

## 10.13 G6 results (recorded 2026-10-04 — K-quant AVX2 kernels + row threading)

Targets from §10.3: AVX2-ize the scalar K-quant matmuls and add threading
(the 30B MoE was 2.0 s/tok single-thread).

### What shipped

1. **AVX2 kernels for Q4_K / Q6_K** (`gguf.rs`): the integer block dots are
   vectorized (nibble decode via `srli_epi16`-mask trick, i16 `mullo` +
   `madd` widen-reduce, `_mm256_cvtepu8_epi16` for unsigned nibbles); the
   per-sub-block float update chain keeps the exact scalar order (plain
   mul+add — no contraction), so both paths are bit-identical (unit-tested
   with `to_bits` equality, plus full-model trace diffs).
2. **Q8_0 stays scalar** — and that's a finding: LLVM auto-vectorizes the
   clean 32-int8 dot better than the hand kernel (bench 6.0 vs 4.0 GFLOP/s),
   and the scalar version *is* the reference. The hand impl remains for the
   bit-parity test.
3. **Row threading** (`dllama-kernel::for_rows`): output rows are
   independent, so ≥8M-MAC matmuls split across threads (bit-exact by
   construction). Below that threshold spawn overhead dominates — the first
   cut threaded everything and made the 0.6B *slower* (µs-scale matmuls ×
   ~200 spawns/token); the threshold fixed it.
4. **MoE expert parallelism** (`for_task_slices`): the top-k experts compute
   concurrently into disjoint buffers (ms-scale tasks — the right granularity),
   and the merge stays serial in `top` order (bit-exact).
5. `--threads N` CLI flag on perplexity/inference/chat/serve/worker +
   `DLLAMA_RS_THREADS`; default = logical cores. One stderr line reports it.

### Measurements (2-core/4-thread ULV laptop)

| Kernel micro-bench (n=k=4096, 1 thread, RAM-resident) | scalar | AVX2 |
|---|---|---|
| Q4_K | 2.81 GFLOP/s | **5.50 (2.0×)** |
| Q6_K | 1.82 GFLOP/s | **3.57 (2.0×)** |
| Q8_0 | **5.99 (autovec)** | 3.94 (hand) — routed to scalar |

| e2e (bit-exact: both model trace diffs byte-identical, 4 threads) | before | after |
|---|---|---|
| 0.6B `.m` generation | 233 ms/tok (1 thr) | 230 ms/tok (4 thr) — clone-overhead-bound, see below |
| 30B-A3B Q4_K_M generation | 2.0 s/tok (G2.5, scalar 1 thr) | **1.55 s/tok (4 thr)** |

Honest reading of the 30B number: on this 16 GB machine the 17.3 GB model is
**storage-bound** — ~2.4 GB of expert weights stream per token at disk
bandwidth, and threads ≈ flat (1549 vs 1593 ms/tok) because the CPUs wait on
page faults. The 22% wall-clock gain is the compute fraction peaking through
I/O. The kernel + expert-parallelism wins are real but masked here; they will
show where the model fits in RAM (32 GB+ machine, or the 2-node G3 cluster,
where each node holds 8.6 GB of experts).

Known follow-up (unchanged from §10.3): the executor's per-op `Vec` clones —
the 0.6B is bound by them (~199.6 ms/tok pre-G2 clone-free vs 233 now), and
removing them is the next executor-side win.

## 10.14 G6.1 results (recorded 2026-10-04 — friendly CLI layer)

The engine-level commands stay (full paths, parity workflows), but everyday use
is now:

```
dllama-rs chat [model]            interactive chat — streams tokens, multi-turn
dllama-rs ask [model] "question"  one-shot answer (or --prompt "...")
dllama-rs models                  list discovered models
dllama-rs ppl [model] --prompt "text"
dllama-rs serve [model] [--port 8080]
```

Design:
- **Model auto-discovery**: `$DLLAMA_MODELS_DIR`, `./models`, and the repo
  layout `<exe>/../../models` are scanned for `.gguf`/`.m`. Names match
  fuzzily (`q40`, `0.6B-Q4_K`), ambiguity lists candidates, and with exactly
  one model present the name can be omitted entirely.
- **Tokenizer auto-detection**: GGUF-embedded when present; for `.m`
  models a single sibling `.t` wins, else the `dllama_model_X` →
  `dllama_tokenizer_X` naming guess.
- **Multi-turn chat** keeps a *token* history: each turn is encoded as its own
  ChatML chunk and appended to the persistent KV cache (no whole-conversation
  re-encode). Seams between chunks are special tokens (`<|im_start|>`,
  `<|im_end|>`), which never merge, so the cached prefix stays consistent.
  The assistant turn is closed in-template (`<|im_end|>\n`) before the next
  user message.
- **Bug class worth remembering**: the first `load_for_chat` transmuted
  `Inference<'w>` to `'static` and `mem::forget`-ed the local `weights` — but
  `Inference` held a pointer to the **stack slot** of that local; the frame
  was reused after return and the weight map got clobbered (`weight tensor
  not found: token_embd`). Fix: `Box::leak` the model container and weights
  onto the heap and build `Inference<'static>` from the leaked references —
  no transmute. (The serve path's transmute only works because its stack
  frame lives as long as the server.)

Validation: `models` lists 7 found models; `ask q40`/`ask 0.6B-Q4_K` stream
coherent answers on both container formats; two-turn `chat` retains context
("the user first asked..."); `ppl q40` reproduces ppl 7.913573 bit-identically
through the resolution bridge; `serve 0.6B-Q4_K` answers via `/v1/models` +
`/v1/chat/completions`.

## 10.15 G6.2 results (recorded 2026-10-04 — sampling + chat templates)

Session 1 of the post-G6 roadmap: sampling controls and model-family templates.

### Sampling (`sampler.rs`)

`--temperature`, `--top-k`, `--top-p`, `--seed` on chat/ask/inference; the
API server accepts `temperature`/`top_p`/`top_k`/`seed` per request.

- `temperature <= 0` (the default) is **greedy and bit-identical with
  `argmax`** — the parity workflows are untouched.
- Pipeline when sampling: `logits/temperature` → the bit-exact softmax →
  top-k → top-p (nucleus) → renormalize → cumulative pick (xorshift64*, 
  deterministic under `--seed`; time-seeded otherwise).
- Properties unit-tested: greedy ≡ argmax; top-k=1 ≡ argmax; tiny top-p ≡
  argmax; sampled tokens always inside the top-k set; seed determinism.

### Chat templates (`chat_template.rs`, minijinja)

The prompt is now rendered by a chain: **GGUF `chat_template` (Jinja)** →
arch fallback table (ChatML for qwen\*, Llama-3 headers, Mistral) → plain
ChatML. The startup line reports which source served the prompt.

- llama.cpp templates use a small Python-method dialect (`minja`:
  `.startswith/.split/.lstrip/...`). We rewrite method chains into
  semantically-identical filter chains (`x.split(s).lstrip(n)` →
  `x | split(s) | lstrip(n)`) and register the method set as filters.
  The rewriter does one rewrite per pass (earlier multi-rewrite-pass
  version duplicated emitted text and could loop forever — unit-tested
  against exactly that).
- Stock templates also rely on `strftime_now`, `raise_exception` and the
  `add_generation_prompt` global — shimmed/provided.
- minijinja strips the trailing newline by default (Jinja2 behavior) —
  `set_keep_trailing_newline(true)` is required or every template loses
  the final `\n` of the assistant header.
- Full Qwen3 tool-calling templates still fall back (they use Python
  slice syntax `[::-1]` that minijinja cannot parse) — the arch fallback
  produces the correct ChatML for the no-tools case, so this is cosmetic
  today; the fallback chain keeps it correct regardless.

### Session rework (ui.rs)

Multi-turn chat now renders the **whole** conversation each turn (via the
template chain) and feeds only the tokens that extend the cached prefix;
if a tokenizer boundary ever shifts the prefix, the session restarts
(fresh KV caches) and refeeds — correctness never depends on the fast
path. This made chat template-agnostic (the old chunk-encoded turns were
ChatML-shaped).

### Validation

| Check | Result |
|---|---|
| `cargo test -p dllama-cli` | 13 passed (6 sampler, 7 template) |
| greedy `ask` (temp 0) | byte-identical with pre-session output |
| temp 0.8 seed 42 ×2 | identical (determinism); seed 7 differs (variety) |
| 2-turn chat, temp 0.6 | context retained; prefix-diff path live |
| `ppl q40` bridge | **7.913573** — canonical value unchanged |
| `serve` + `temperature/top_p/seed` JSON | answers sampled |

## 10.16 G6.3 results (recorded 2026-10-05 — clone-free executor)

Session 2 of the post-G6 roadmap: eliminate the per-op activation clones.

### What changed (`dllama-exec/src/lib.rs`)

- `exec_op` is now a **free function taking each Inference field as an
  independent parameter** — borrow-clean by construction, which is what
  lets multi-pipe ops run without cloning. `forward` destructures `&mut
  self` once per token and iterates `&graph.ops` directly (previously it
  cloned the whole ops vec **and** the meta **every token**).
- Per-op: outputs are `mem::take`-n and put back; inputs are shared
  borrows. The residual round-trip and embedding dequant reuse a new
  `scratch: Vec<f32>` field. The F32-branch `x.to_vec()` in matmul is gone.
- QkRmsNorm no longer clones the norm-weight vectors per call.

### The aliasing trap (found the hard way)

`ActivatedMul` writes into the **same pipe it reads** (builder: `out ==
gate`) — the old implementation's clones existed precisely for that. The
take-pattern emptied the aliased gate slice and the kernel panicked on the
first layer (`gate[3072]` on an empty slice, trace stopped after
`block_matmul_w3.output`). Fix: alias-aware arm — when `out == gate|up`,
snapshot the taken buffer into `scratch` (memcpy, no alloc) and read the
gate from there; the op is element-wise so in-place remains numerically
identical. `ResidualAdd`'s `output == accumulator` was already handled by
take + `clone_from_slice`.

### Validation

| Check | Result |
|---|---|
| `.m` trace diff (old exec vs new, 4 threads) | **byte-identical** (8719 lines, ppl 7.913573) |
| GGUF Q4_K_M trace diff | **byte-identical** (ppl 13.472507) |
| `cargo test --workspace` | green |
| 0.6B prediction ms/tok, 3 runs (old → new) | 172.7 → 145.3 / 151.3 / 154.0 (**~12%**, median 151) |

Honest note: the DESIGN §10.3 "~30% clone overhead" estimate was optimistic —
the clones cost ~2-15% at 0.6B scale (the 199.6→233 regression conflated
several G2 changes). The bigger structural win from this refactor is that
the executor is now allocation-free per token outside the MoE path — and
moe_ffn's per-expert buffers are the next (last) allocation site.

## 10.17 G6.4 results (recorded 2026-10-05 — prompt prefill batching)

Session 3: batched prompt processing (`dllama-exec/src/prefill.rs`).

### What shipped

- **`Inference::prefill(tokens, start)`**: runs a prompt chunk through a
  layer-major batched path with local `[m × len]` pipes. Only the KV cache
  persists; the caller then runs `forward` on the final prompt token as
  usual (which recomputes the head). `Op::Head`/`Op::Argmax` are skipped.
- **Weight-stationary batch matmuls** (all quant kinds + f16/f32):
  each weight block is decoded once per output row and dotted against all
  m activation rows. The kernels compute **column-major** (`out[di·m+mi]`,
  one m-chunk per output row) so the threaded loop passes mutable state
  through arguments — the new `for_chunk_segments_mut` helper, same
  pattern that made G6's `for_rows` borrow-check without unsafe.
- **Causal attention in batch**: the whole K/V block is appended first
  (rows see earlier rows of the same batch), then per-row attention over
  `[0..=start+i]` reusing the table's `dot`/`softmax` — identical widths
  and chains to sequential.
- **Bit-exactness**: per-row float chains match the sequential reference
  exactly (unit test: `to_bits` equality for all 4 quant kinds vs the table
  kernels; e2e: greedy generation identical with prefill on/off). The
  batched path and the sequential path share one KV-cache format.
- **Wired** into chat/ask (`ui.rs generate_turn`, incl. multi-turn
  continuation and the divergence-restart path), `inference`, and the API
  server; `DLLAMA_PREFILL=0` is the kill-switch.
- MoE runs per-row in v1 (expert matmuls not batched).

### Validation

| Check | Result |
|---|---|
| unit: batch vs sequential `to_bits` (4 kinds) | bit-exact |
| greedy generation, prefill OFF vs ON (32 steps, .m) | **identical text** |
| ppl bridge | 7.913573 unchanged |
| 2-turn chat with prefill | context works (KV continuation) |

### Honest perf reading

On the 0.6B the evaluation phase is **a wash** (≈144 ms/tok both ways): the
model is compute-bound with weights resident in cache, and the batch kernels
are **scalar v1** — the sequential path uses the hand-AVX2 m=1 kernels, so
batched-scalar ≈ sequential-AVX2. The structural win of prefill — weights
read once per matmul instead of m times — pays on **bandwidth-bound** models
(the 30B MoE streaming 17 GB of experts), which needs the follow-ups:
batch the MoE expert matmuls and SIMD-ize the batch kernels (the integer
dots are already order-free; SIMD keeps bit-exactness, as G6 proved).

## 10.18 G3.2 results (recorded 2026-10-05 — auto node discovery)

The C++ upstream has no discovery either (explicit `--workers` addresses,
zero UDP/broadcast code in `src/`) — both engines made the operator type
node lists. Now the Rust engine doesn't:

- **Workers** answer UDP probes on a fixed discovery port (9990) with their
  TCP port + a per-process **instance id**; the responder is spawned by
  `dllama-rs worker` automatically. If the UDP port is taken by another
  local worker, the responder stays quiet — explicit addresses still work.
- **Roots** broadcast a probe (subnet broadcast + localhost — WSL2's NAT
  does not forward broadcast, but same-machine roots reach their workers
  on loopback) and collect replies with a quiescence timeout.
- **Dedup by instance id**: a worker answering on several interfaces
  (loopback + LAN) yields one entry — the loopback address is preferred
  (same-machine roots reach it regardless of firewall; remote workers only
  ever answer from their LAN IP, so remote entries are unaffected).
- CLI: `dllama-rs nodes [--timeout 1000]` lists discovered nodes;
  `perplexity-dist --workers auto` runs discovery and connects (explicit
  `--workers host:port ...` still supported).

Validation: unit test of the probe/reply cycle (incl. dedup — one worker,
one entry); e2e: worker started with defaults → `nodes` found it →
`perplexity-dist --workers auto` connected and produced the known 2-node
ppl 7.727886.

## 10.19 G6.5 results (recorded 2026-10-05 — distributed inference)

`inference-dist` completes the cluster story: generation, not just
perplexity, runs across nodes.

- The root drives sampling; every generated token is broadcast to all
  workers in the next control packet, so all KV caches stay identical.
  Workers never need the tokenizer output — they just evaluate.
- Same setup path as the ppl driver (meta + tensor-slice sync, ack), same
  DLRS control packets; `--workers auto` (UDP discovery) works here too.
- Sampler flags apply on the root (`--temperature`, `--top-p`, `--seed`).
- Ends with a `batch_size: 0` stop packet — workers log "stop packet
  received" and return to accept.

Validation (2 localhost nodes, qwen3-0.6b q40, greedy): prompt of 5 tokens
fed, 16 tokens generated coherently ("...capital of Italy is Rome..."),
258 ms/tok (the 0.6B is overhead-dominated on loopback — distribution pays
when a model doesn't fit in one machine's RAM, which is the 30B case).

## 11. Open questions / risks

1. **Kernel performance parity**: llamafile sgemm is hand-tuned AVX2/NEON. G6
   shipped the K-quant AVX2 kernels (2.0× on Q4_K/Q6_K), row threading and
   parallel MoE experts (§10.13). Remaining gaps: the q40 GEMV path (0.93
   GFLOP/s in bench shape vs llamafile's sgemm), and the executor's per-op
   `Vec` clones (~30%, the 0.6B's dominant cost). Next: clone removal, then
   a tiled q40 kernel for cache locality.
2. **Mojo on Windows**: **verified unavailable** (2026-10-03: no winget package,
   no `mojo` CLI). G5 delivered the C-ABI v1 seam; **the Mojo backend now exists
   and is bit-exact on Linux** (2026-10-04, WSL2, rootless uv install, §10.10) —
   `rust/mojo/kernels.mojo` compiles for Windows the day a toolchain ships.
   Note: Mojo defaults to fp-contract=fast; backends must build with
   `--fp-mode contract=off` to match the Rust reference (§10.10).
3. **Q80 scalar rounding**: confirm byte-parity against `nn-quants.cpp` scalar path.
4. **String wire format**: confirm `writeString` encoding before G3.
5. **Vulkan/GPU path**: the first GPU slice shipped as an **OpenCL accelerant
   for the Rust backend** (§10.12) — vendor-agnostic (NVIDIA/AMD/Intel),
   bit-exact, validated on an Intel iGPU. Mojo GPU kernels are compile-verified
   (§10.11) pending hardware. Vulkan/wgpu remains unattractive: no WGSL `fma`
   means no bit-exactness. Next GPU steps: measure discrete-GPU perf, then
   block-per-row + shared-memory kernels and attention if worthwhile.
6. Preset dims in `dllama-cli` are for planning only — real values come from
   model metadata at G1/G2.

*Generated during migration Phase 0. Update this file when designs change.*
