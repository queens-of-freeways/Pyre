//! dllama-rs — CLI for the Rust engine (Phase 0: benchmarks + graph inspection).
//!
//! Flag style mirrors the C++ binary (`--nthreads 4`, `--nodes 4`) so muscle
//! memory transfers during migration.

use dllama_ir::{build_decoder, check_sharding, per_token_sync_bytes, shard_all, FloatType, Graph, ModelMeta, Op, SyncKind};
use std::time::Instant;

mod api;
mod chat_template;
mod g1;
mod sampler;
mod ui;

// ---------------------------------------------------------------------------
// Model presets (planning values only — real values come from model metadata
// at G1/G2; see DESIGN.md §11.6)
// ---------------------------------------------------------------------------

fn llama_3_1_8b() -> ModelMeta {
    ModelMeta {
        name: "llama-3.1-8b".into(),
        arch: dllama_ir::Arch::Llama,
        dim: 4096,
        hidden_dim: 14336,
        moe_hidden_dim: 0,
        n_layers: 32,
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 128,
        n_experts: 0,
        n_active_experts: 0,
        vocab_size: 128256,
        seq_len: 4096,
        orig_seq_len: 131072,
        hidden_act: dllama_ir::HiddenAct::Silu,
        norm_epsilon: 1e-5,
        rope_theta: 500000.0,
        rope_type: dllama_ir::RopeType::Llama31,
        rope_scaling: Some(dllama_ir::RopeScaling {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            orig_max_seq_len: 8192,
        }),
        qk_norm: false,
        weight_type: FloatType::Q40,
        sync_type: FloatType::Q80,
    }
}

fn qwen3_8b() -> ModelMeta {
    let mut m = llama_3_1_8b();
    m.name = "qwen3-8b".into();
    m.arch = dllama_ir::Arch::Qwen3;
    m.hidden_dim = 12288;
    m.n_layers = 36;
    m.vocab_size = 151936;
    m.rope_theta = 1_000_000.0;
    m.rope_type = dllama_ir::RopeType::Falcon;
    m.rope_scaling = None;
    m.qk_norm = true;
    m.norm_epsilon = 1e-6;
    m
}

fn qwen3_30b_a3b() -> ModelMeta {
    let mut m = qwen3_8b();
    m.name = "qwen3-30b-a3b".into();
    m.arch = dllama_ir::Arch::Qwen3Moe;
    m.dim = 2048;
    m.hidden_dim = 0;
    m.moe_hidden_dim = 768;
    m.n_layers = 48;
    m.n_heads = 32;
    m.n_kv_heads = 4;
    m.n_experts = 128;
    m.n_active_experts = 8;
    m
}

fn preset(name: &str) -> Option<ModelMeta> {
    match name {
        "llama-3.1-8b" | "llama8b" => Some(llama_3_1_8b()),
        "qwen3-8b" => Some(qwen3_8b()),
        "qwen3-30b-a3b" => Some(qwen3_30b_a3b()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// graph-dump
// ---------------------------------------------------------------------------

fn print_op(index: usize, op: &Op) {
    let detail = match op {
        Op::Embedding { w, out, .. } => format!("{w:24} -> pipe{out}"),
        Op::MatMul { w, parallel, input, output } => {
            let p = match parallel {
                dllama_ir::Parallel::OutSharded => "out-sharded",
                dllama_ir::Parallel::InSharded => "in-sharded",
                dllama_ir::Parallel::Replicated => "replicated",
            };
            format!("{w:24} [{p}] pipe{input} -> pipe{output}")
        }
        Op::RmsNorm { w, input, output, .. } => format!("{w:24} pipe{input} -> pipe{output}"),
        Op::QkRmsNorm { qw, q_pipe, k_pipe, .. } => format!("{qw:24} q=pipe{q_pipe} k=pipe{k_pipe}"),
        Op::Rope { q_pipe, k_pipe, .. } => format!("pipe{q_pipe}, pipe{k_pipe}"),
        Op::Attention { layer, output, .. } => format!("layer {layer:3} -> pipe{output}"),
        Op::ActivatedMul { gate, up, out, .. } => format!("pipe{gate} x pipe{up} -> pipe{out}"),
        Op::Moe { layer, n_experts, n_active, output, .. } => {
            format!("layer {layer:3} {n_active}/{n_experts} experts -> pipe{output}")
        }
        Op::ResidualAdd { accumulator, addend, output } => {
            format!("pipe{accumulator} += pipe{addend} -> pipe{output}")
        }
        Op::Head { w, input, output } => format!("{w:24} pipe{input} -> pipe{output}"),
        Op::Sync { kind, pipe, dtype } => {
            let k = match kind {
                SyncKind::AllReduce => "ALL-REDUCE",
                SyncKind::AllGather => "ALL-GATHER",
            };
            format!("{k:11} pipe{pipe} ({})", dtype.name())
        }
        Op::Softmax { input, output } | Op::Argmax { input, output } => {
            format!("pipe{input} -> pipe{output}")
        }
    };
    let marker = if matches!(op, Op::Sync { .. }) { "  \u{1f4e1}" } else { "" };
    println!("  [{index:03}] {:<12} {detail}{marker}", op.kind());
}

fn cmd_graph_dump(args: &[String]) -> i32 {
    let model = flag(args, "--model").unwrap_or("llama-3.1-8b");
    let nodes: u32 = flag(args, "--nodes").unwrap_or("1").parse().unwrap_or(1);

    let meta = match preset(model) {
        Some(m) => m,
        None => {
            eprintln!("unknown model preset '{model}' (llama-3.1-8b | qwen3-8b | qwen3-30b-a3b)");
            return 1;
        }
    };

    println!("\n\u{1f9e0} Model: {} (preset)", meta.name);
    println!(
        "   dim={} layers={} heads={} kv-heads={} head-dim={} vocab={} ffn={} seq={}",
        meta.dim,
        meta.n_layers,
        meta.n_heads,
        meta.n_kv_heads,
        meta.head_dim_or_derived(),
        meta.vocab_size,
        if meta.is_moe() { format!("{} experts x {}", meta.n_experts, meta.moe_hidden_dim) } else { meta.hidden_dim.to_string() },
        meta.seq_len,
    );
    println!(
        "   weights={} sync={} rope-theta={}{}",
        meta.weight_type.name(),
        meta.sync_type.name(),
        meta.rope_theta,
        if meta.qk_norm { " qk-norm" } else { "" },
    );

    if let Err(e) = check_sharding(&meta, nodes) {
        println!("\n\u{1f6a8} Sharding check FAILED for {nodes} nodes: {e}");
        return 1;
    }

    let graph: Graph = build_decoder(&meta, nodes);
    println!("\n\u{1f5c3} Graph: {} ops, {} syncs, {} pipes", graph.ops.len(), graph.count_syncs(), graph.pipes.len());

    let max_show = flag(args, "--show").unwrap_or("40".into()).parse::<usize>().unwrap_or(40);
    println!("\n   (first {max_show} ops)");
    for (i, op) in graph.ops.iter().take(max_show).enumerate() {
        print_op(i, op);
    }
    if graph.ops.len() > max_show {
        println!("   ... {} more ops", graph.ops.len() - max_show);
    }

    if nodes > 1 {
        println!("\n\u{1f5fd} Sharding plan ({nodes} nodes):");
        for s in shard_all(&meta, nodes).unwrap() {
            let role = if s.node_index == 0 { "ROOT " } else { "worker" };
            println!(
                "   node {} ({role}): q-rows {} ({} heads) kv-rows {} ({} kv-heads) ffn-rows {} head-rows {}",
                s.node_index, s.q_rows, s.n_heads, s.kv_rows, s.n_kv_heads, s.ffn_rows, s.cls_rows,
            );
        }
        let b = per_token_sync_bytes(&meta, nodes);
        println!(
            "\n   estimated sync traffic: {:.2} MB/token/worker ({})",
            b as f64 / 1024.0 / 1024.0,
            meta.sync_type.name(),
        );
    } else {
        println!("\n   single node: no sharding, no sync ops");
    }
    println!();
    0
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

fn bench_matmul(args: &[String]) -> i32 {
    let m: usize = flag(args, "--m").unwrap_or("1".into()).parse().unwrap_or(1);
    let n: usize = flag(args, "--n").unwrap_or("4096".into()).parse().unwrap_or(4096);
    let k: usize = flag(args, "--k").unwrap_or("4096".into()).parse().unwrap_or(4096);
    let iters: usize = flag(args, "--iters").unwrap_or("10".into()).parse().unwrap_or(10);

    println!("\n\u{1f3c3} dllama-rs matmul benchmark (single-thread, backend: cpu-rust)");
    println!("   m={m} n={n} k={k} iters={iters} (+2 warmup)");

    // deterministic pseudo-random data
    let x: Vec<f32> = (0..m * k).map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0).collect();
    let w_f32: Vec<f32> = (0..n * k).map(|i| ((i * 40503) % 997) as f32 / 500.0 - 1.0).collect();

    // pack weights to q40 rows (dllama layout: [n][k] rows, blocks within row)
    let row_bytes = dllama_quant::q40_bytes(k);
    let mut w_q40 = vec![0u8; n * row_bytes];
    for ni in 0..n {
        let mut tmp = vec![0u8; row_bytes];
        dllama_quant::quantize_row_q40(&w_f32[ni * k..][..k], &mut tmp);
        w_q40[ni * row_bytes..][..row_bytes].copy_from_slice(&tmp);
    }

    let mut out = vec![0.0f32; m * n];

    let run = |label: &str, weight_bytes: usize, f: &mut dyn FnMut()| {
        f(); // warmup x2
        f();
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        let dt = t0.elapsed().as_secs_f64() / iters as f64;
        let gflops = 2.0 * (m * n * k) as f64 / dt / 1e9;
        let gbps = weight_bytes as f64 / dt / 1e9;
        println!("   {label:<18} {dt:>8.3} s/iter   {gflops:6.2} GFLOP/s   {gbps:6.2} GB/s weights");
    };

    run("f32", n * k * 4, &mut || {
        dllama_kernel::cpu::matmul_f32(&mut out, &x, &w_f32, m, n, k)
    });
    run("q40 (on-the-fly)", n * row_bytes, &mut || {
        dllama_kernel::cpu::matmul_q40(&mut out, &x, &w_q40, m, n, k)
    });
    // predequant upper bound
    let w_deq: Vec<f32> = (0..n)
        .flat_map(|ni| dllama_quant::dequantize_q40_row(&w_q40[ni * row_bytes..], k))
        .collect();
    run("q40 (predequant)", n * row_bytes, &mut || {
        dllama_kernel::cpu::matmul_f32(&mut out, &x, &w_deq, m, n, k)
    });

    // G6: q80-input K-quant kernels — scalar reference vs AVX2 (RAM-resident,
    // single-thread: isolates the kernel speedup from threading and I/O).
    if k % 256 == 0 {
        dllama_kernel::set_threads(1);
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut byte = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as u8
        };
        let xk: Vec<f32> = (0..k).map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0).collect();
        let mut xq80 = vec![0u8; dllama_quant::q80_bytes(k)];
        dllama_quant::quantize_row_q80(&xk, &mut xq80);
        for (kind_name, kind, rbytes) in [
            ("Q4_K", dllama_quant::QuantKind::GgufQ4K, 144usize),
            ("Q6_K", dllama_quant::QuantKind::GgufQ6K, 210),
            ("Q8_0", dllama_quant::QuantKind::GgufQ8_0, 34),
        ] {
            let blocks = if rbytes == 34 { k / 32 } else { k / 256 };
            let scale_off = if rbytes == 210 { 208 } else { 0 };
            let rb = blocks * rbytes;
            let mut wq = vec![0u8; n * rb];
            for ni in 0..n {
                for b in wq[ni * rb..(ni + 1) * rb].iter_mut() {
                    *b = byte();
                }
                let row = &mut wq[ni * rb..(ni + 1) * rb];
                for blk in 0..blocks {
                    let at = blk * rbytes + scale_off;
                    row[at] = 0x1F;
                    row[at + 1] = 0x21; // f16 0.01
                    if rbytes == 144 {
                        row[at + 2] = 0x14; // f16 ~0.002 (dmin)
                        row[at + 3] = 0x21;
                    }
                }
            }
            let out_k = &mut out[..n];
            run(&format!("{kind_name} (scalar)"), n * rb, &mut || match kind {
                dllama_quant::QuantKind::GgufQ4K => dllama_kernel::gguf::matmul_q80_q4_k(out_k, &xq80, &wq, n, k),
                dllama_quant::QuantKind::GgufQ6K => dllama_kernel::gguf::matmul_q80_q6_k(out_k, &xq80, &wq, n, k),
                _ => dllama_kernel::gguf::matmul_q80_q8_0(out_k, &xq80, &wq, n, k),
            });
            run(&format!("{kind_name} (avx2)"), n * rb, &mut || {
                dllama_kernel::gguf::matmul_q80(out_k, &xq80, kind, &wq, n, k)
            });
        }
    }
    println!();
    0
}

fn bench_quant(args: &[String]) -> i32 {
    let n: usize = flag(args, "--n").unwrap_or("1048576".into()).parse().unwrap_or(1 << 20);
    let iters: usize = flag(args, "--iters").unwrap_or("50".into()).parse().unwrap_or(50);
    let x: Vec<f32> = (0..n).map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0).collect();

    println!("\n\u{1f3c3} dllama-rs quant benchmark ({n} values/iter, {iters} iters)");
    let mut q80 = vec![0u8; dllama_quant::q80_bytes(n)];
    let t0 = Instant::now();
    for _ in 0..iters {
        dllama_quant::quantize_row_q80(&x, &mut q80);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    println!("   quantize q80: {dt:.3} ms/iter  {:.2} M values/s  {:.2} GB/s in", n as f64 / dt / 1e6, (n * 4) as f64 / dt / 1e9);

    let mut back = vec![0.0f32; n];
    let t0 = Instant::now();
    for _ in 0..iters {
        dllama_quant::dequantize_row_q80(&q80, &mut back, n);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    println!("   dequant q80: {dt:.3} ms/iter  {:.2} M values/s", n as f64 / dt / 1e6);

    let mut q40 = vec![0u8; dllama_quant::q40_bytes(n)];
    let t0 = Instant::now();
    for _ in 0..iters {
        dllama_quant::quantize_row_q40(&x, &mut q40);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    println!("   quantize q40: {dt:.3} ms/iter  {:.2} M values/s", n as f64 / dt / 1e6);
    println!();
    0
}

// ---------------------------------------------------------------------------
// arg parsing (dllama-style --key value)
// ---------------------------------------------------------------------------

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            return it.next().map(|s| s.as_str());
        }
    }
    None
}

fn help() -> i32 {
    println!("dllama-rs — Distributed Llama, Rust engine");
    println!();
    println!("EASY (models are found automatically in ./models or $DLLAMA_MODELS_DIR):");
    println!("  dllama-rs chat [model]            interactive chat (streams tokens, multi-turn)");
    println!("  dllama-rs ask [model] \"question\"  one-shot answer (also: --prompt \"...\")");
    println!("  dllama-rs models                  list discovered models");
    println!("  dllama-rs ppl [model] --prompt \"text\"   perplexity of a text");
    println!("  dllama-rs serve [model] [--port 8080]    OpenAI-style /v1 API server");
    println!();
    println!("  model names are fuzzy: `chat qwen3` matches dllama_model_qwen3_0.6b_q40.m.");
    println!("  With a single model present the name can be omitted entirely.");
    println!("  Useful flags: --threads N | --steps N | --max-seq-len N |");
    println!("                --temperature 0.7 --top-k 40 --top-p 0.9 --seed N --system \"...\"");
    println!("                (temperature 0 = greedy, the deterministic default)");
    println!("  Backends: DLLAMA_KERNEL_LIB=<plugin>, DLLAMA_RUST_GPU=0, DLLAMA_RS_THREADS=N");    println!();
    println!("ADVANCED (full paths, engine-level control):");
    println!("  dllama-rs perplexity --model <f> --tokenizer <f> --prompt <text> [--max-seq-len 4096]");
    println!("  dllama-rs inference  --model <f> --tokenizer <f> --prompt <text> --steps 64");
    println!("  dllama-rs chat       --model <f> [--tokenizer <f>]");
    println!("  dllama-rs gguf-check --model <f>   | gguf-to-m | m-to-gguf");
    println!("  dllama-rs perplexity-dist --model <f> --workers auto --prompt <text>   (or: --workers host:port ...)
  dllama-rs nodes [--timeout 1000]        list discovered cluster nodes");
    println!("  dllama-rs bench matmul [--n 4096] [--k 4096]   | bench quant");
    println!("  dllama-rs graph-dump --model <preset> --nodes <n>");
    println!();
    println!("See rust/DESIGN.md for the architecture, gates G0-G7 and parity notes.");
    0
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = if args.is_empty() {
        help()
    } else {
        match args[0].as_str() {
            "chat" | "run" => ui::cmd_chat(&args[1..]),
            "ask" => ui::cmd_ask(&args[1..]),
            "models" => ui::cmd_models(),
            "ppl" => ui::cmd_ppl(&args[1..]),
            "graph-dump" => cmd_graph_dump(&args[1..]),
            "bench" => match args.get(1).map(|s| s.as_str()) {
                Some("matmul") => bench_matmul(&args[2..]),
                Some("quant") => bench_quant(&args[2..]),
                _ => help(),
            },
            "perplexity" => g1::run_perplexity(&args[1..]),
            "inference" => g1::run_inference(&args[1..]),
            "gguf-check" => g1::run_gguf_check(&args[1..]),
            "gguf-to-m" => g1::run_gguf_to_m(&args[1..]),
            "m-to-gguf" => g1::run_m_to_gguf(&args[1..]),
            "perplexity-dist" => g1::run_distributed_perplexity(&args[1..]),
            "chat" => g1::run_chat(&args[1..]),
            "serve" => {
                g1::apply_threads_flag(&args[1..]);
                let host = flag(&args[1..], "--host").unwrap_or("127.0.0.1").to_string();
                let port: u16 = flag(&args[1..], "--port").unwrap_or("8080").parse().unwrap_or(8080);
                let model = match ui::resolve_from_args(&args[1..], ui::SERVE_FLAGS) {
                    Ok(m) => m,
                    Err(e) => { eprintln!("{e}"); std::process::exit(1); }
                };
                let mp = model.path.to_string_lossy().into_owned();
                let source = match g1::open_model(&mp, 4096) {
                    Ok(f) => f,
                    Err(e) => { eprintln!("error: {e}"); std::process::exit(1); }
                };
                let meta = match source.meta(4096) {
                    Ok(m) => m,
                    Err(e) => { eprintln!("error: {e}"); std::process::exit(1); }
                };
                let weights = match source.weights() {
                    Ok(w) => w,
                    Err(e) => { eprintln!("error: {e}"); std::process::exit(1); }
                };
                let tok_path = match ui::find_tokenizer(&model.path) {
                    Ok(t) => t,
                    Err(e) => { eprintln!("error: {e}"); std::process::exit(1); }
                };
                let tok_str = tok_path.map(|p| p.to_string_lossy().into_owned());
                let tokenizer = match source.tokenizer(tok_str.as_deref()) {
                    Ok(t) => t,
                    Err(e) => { eprintln!("error: {e}"); std::process::exit(1); }
                };
                let static_weights: &'static dllama_model::Weights<'static> =
                    unsafe { std::mem::transmute::<&dllama_model::Weights<'_>, &'static dllama_model::Weights<'static>>(&weights) };
                let rt = tokio::runtime::Runtime::new().unwrap();
                if let Err(e) = rt.block_on(api::serve_api(meta, tokenizer, static_weights, &host, port)) {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
                0
            }
            "nodes" => {
                // G3.2: auto node discovery (UDP probe/reply on port 9990)
                let timeout: u64 = flag(&args[1..], "--timeout")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1000);
                match dllama_cluster::discover_nodes(timeout) {
                    Ok(list) if list.is_empty() => {
                        eprintln!("no nodes found — start workers (dllama-rs worker) and retry");
                        1
                    }
                    Ok(list) => {
                        println!("Nodes found ({}):", list.len());
                        for n in &list {
                            println!("  {n}");
                        }
                        println!("\nusage: dllama-rs perplexity-dist --model <f> --workers auto --prompt <text>");
                        0
                    }
                    Err(e) => {
                        eprintln!("discovery failed: {e}");
                        1
                    }
                }
            }
            "worker" => {
                g1::apply_threads_flag(&args[1..]);
                let host = flag(&args[1..], "--host").unwrap_or("127.0.0.1").to_string();
                let port: u16 = flag(&args[1..], "--port").unwrap_or("9998").parse().unwrap_or(9998);
                match dllama_cluster::serve_worker(&host, port) {
                    Ok(()) => 0,
                    Err(e) => {
                        eprintln!("🚨 {e}");
                        1
                    }
                }
            }
            "help" | "--help" | "-h" => help(),
            other => {
                eprintln!("unknown command '{other}'");
                help()
            }
        }
    };
    std::process::exit(code);
}

