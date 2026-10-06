//! dllama-cluster: G3 distributed runtime — TCP tensor-parallel cluster.
//!
//! Wire protocol v1 (all LE, blocking std::net):
//!
//! ```text
//! setup (root -> worker):
//!   "DLRS" | u32 version=1
//!   ModelMeta (fixed layout)
//!   u32 node_index | u32 n_nodes
//!   u32 n_tensors; per: string name, u8 kind, u64 len, bytes
//!   worker -> root: u32 ACK
//!
//! per token (root -> worker):
//!   { u32 token, u32 position, u32 batch_size }  // batch_size 0 = stop
//!
//! SyncKind::AllReduce (C++ SYNC_NODE_SLICES): all-gather + local merge
//!   worker -> root: u64 len, q80 bytes (its partial)
//!   root -> worker: u32 n_slices; per slice: u64 len, q80 bytes
//!   every node: pipe = sum(dequant(slice_i))
//!
//! SyncKind::AllGather (head): worker -> root: u64 len, f32 slice
//! ```

use dllama_ir::shard::{shard_node, NodeShard, Slice};
use dllama_ir::{Arch, FloatType, HiddenAct, ModelMeta, RopeScaling, RopeType};
use dllama_model::{QuantKind, Tensor, Weights};
use dllama_quant::{dequantize_row_q80, q80_bytes, quantize_row_q80};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

pub const MAGIC: [u8; 4] = *b"DLRS";
pub const PROTOCOL_VERSION: u32 = 1;
pub const ACK: u32 = 0xD1CE;

// ---------------------------------------------------------------------------
// control packet
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlPacket {
    pub token: u32,
    pub position: u32,
    pub batch_size: u32,
}

impl ControlPacket {
    pub const STOP: ControlPacket = ControlPacket { token: 0, position: 0, batch_size: 0 };

    pub fn encode(&self) -> [u8; 12] {
        let mut b = [0u8; 12];
        b[..4].copy_from_slice(&self.token.to_le_bytes());
        b[4..8].copy_from_slice(&self.position.to_le_bytes());
        b[8..].copy_from_slice(&self.batch_size.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.len() < 12 {
            return None;
        }
        Some(Self {
            token: u32::from_le_bytes(b[..4].try_into().ok()?),
            position: u32::from_le_bytes(b[4..8].try_into().ok()?),
            batch_size: u32::from_le_bytes(b[8..].try_into().ok()?),
        })
    }
}

// ---------------------------------------------------------------------------
// stream helpers
// ---------------------------------------------------------------------------

trait WireExt: Read + Write {
    fn w_u32(&mut self, v: u32) -> Result<(), String> {
        self.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
    }
    fn r_u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u32::from_le_bytes(b))
    }
    fn w_u64(&mut self, v: u64) -> Result<(), String> {
        self.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
    }
    fn r_u64(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u64::from_le_bytes(b))
    }
    fn w_f32(&mut self, v: f32) -> Result<(), String> {
        self.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
    }
    fn r_f32(&mut self) -> Result<f32, String> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(f32::from_le_bytes(b))
    }
    fn w_str(&mut self, s: &str) -> Result<(), String> {
        self.w_u64(s.len() as u64)?;
        self.write_all(s.as_bytes()).map_err(|e| e.to_string())
    }
    fn r_str(&mut self) -> Result<String, String> {
        let len = self.r_u64()? as usize;
        let mut b = vec![0u8; len];
        self.read_exact(&mut b).map_err(|e| e.to_string())?;
        String::from_utf8(b).map_err(|e| e.to_string())
    }
    fn w_blob(&mut self, v: &[u8]) -> Result<(), String> {
        self.w_u64(v.len() as u64)?;
        self.write_all(v).map_err(|e| e.to_string())
    }
    fn r_blob(&mut self) -> Result<Vec<u8>, String> {
        let len = self.r_u64()? as usize;
        let mut b = vec![0u8; len];
        self.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(b)
    }
}

impl WireExt for TcpStream {}

// ---------------------------------------------------------------------------
// ModelMeta wire format
// ---------------------------------------------------------------------------

pub fn write_meta(w: &mut TcpStream, m: &ModelMeta) -> Result<(), String> {
    w.w_str(&m.name)?;
    w.w_u32(match m.arch {
        Arch::Llama => 0,
        Arch::Qwen3 => 1,
        Arch::Qwen3Moe => 2,
    })?;
    w.w_u32(m.dim)?;
    w.w_u32(m.hidden_dim)?;
    w.w_u32(m.moe_hidden_dim)?;
    w.w_u32(m.n_layers)?;
    w.w_u32(m.n_heads)?;
    w.w_u32(m.n_kv_heads)?;
    w.w_u32(m.head_dim)?;
    w.w_u32(m.n_experts)?;
    w.w_u32(m.n_active_experts)?;
    w.w_u32(m.vocab_size)?;
    w.w_u32(m.seq_len)?;
    w.w_u32(m.orig_seq_len)?;
    w.w_u32(match m.hidden_act {
        HiddenAct::Silu => 1,
        HiddenAct::Gelu => 0,
    })?;
    w.w_f32(m.norm_epsilon)?;
    w.w_f32(m.rope_theta)?;
    w.w_u32(m.rope_type.wire() as u32)?;
    match &m.rope_scaling {
        Some(s) => {
            w.w_u32(1)?;
            w.w_f32(s.factor)?;
            w.w_f32(s.low_freq_factor)?;
            w.w_f32(s.high_freq_factor)?;
            w.w_u32(s.orig_max_seq_len)?;
        }
        None => w.w_u32(0)?,
    }
    w.w_u32(m.qk_norm as u32)?;
    w.w_u32(m.weight_type.wire())?;
    w.w_u32(m.sync_type.wire())?;
    Ok(())
}

pub fn read_meta(r: &mut TcpStream) -> Result<ModelMeta, String> {
    let name = r.r_str()?;
    let arch = match r.r_u32()? {
        0 => Arch::Llama,
        1 => Arch::Qwen3,
        2 => Arch::Qwen3Moe,
        other => return Err(format!("bad arch id {other}")),
    };
    let dim = r.r_u32()?;
    let hidden_dim = r.r_u32()?;
    let moe_hidden_dim = r.r_u32()?;
    let n_layers = r.r_u32()?;
    let n_heads = r.r_u32()?;
    let n_kv_heads = r.r_u32()?;
    let head_dim = r.r_u32()?;
    let n_experts = r.r_u32()?;
    let n_active_experts = r.r_u32()?;
    let vocab_size = r.r_u32()?;
    let seq_len = r.r_u32()?;
    let orig_seq_len = r.r_u32()?;
    let hidden_act = match r.r_u32()? {
        1 => HiddenAct::Silu,
        _ => HiddenAct::Gelu,
    };
    let norm_epsilon = r.r_f32()?;
    let rope_theta = r.r_f32()?;
    let rope_type = RopeType::from_wire(r.r_u32()? as i32).ok_or("bad rope type")?;
    let rope_scaling = if r.r_u32()? == 1 {
        Some(RopeScaling {
            factor: r.r_f32()?,
            low_freq_factor: r.r_f32()?,
            high_freq_factor: r.r_f32()?,
            orig_max_seq_len: r.r_u32()?,
        })
    } else {
        None
    };
    let qk_norm = r.r_u32()? == 1;
    let weight_type = FloatType::from_wire(r.r_u32()?).ok_or("bad weight type")?;
    let sync_type = FloatType::from_wire(r.r_u32()?).ok_or("bad sync type")?;
    Ok(ModelMeta {
        name,
        arch,
        dim,
        hidden_dim,
        moe_hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        n_experts,
        n_active_experts,
        vocab_size,
        seq_len,
        orig_seq_len,
        hidden_act,
        norm_epsilon,
        rope_theta,
        rope_type,
        rope_scaling,
        qk_norm,
        weight_type,
        sync_type,
    })
}

// ---------------------------------------------------------------------------
// tensor sharding (root side)
// ---------------------------------------------------------------------------

/// Column-shard a [rows][k] tensor: repack contiguous per-row column blocks.
fn repack_cols(t: &Tensor, cols: &Slice, k: usize, n_rows: usize) -> Result<Vec<u8>, String> {
    let block = t.kind.block_elems();
    let col_len = (cols.end - cols.start) as usize;
    if col_len % block != 0 || cols.start as usize % block != 0 {
        return Err(format!(
            "column shard {}..{} not aligned to {}-value blocks ({:?})",
            cols.start, cols.end, block, t.kind
        ));
    }
    let row_bytes_full = t.kind.row_bytes(k) as usize;
    let row_bytes_slice = t.kind.row_bytes(col_len) as usize;
    let bb = t.kind.block_bytes();
    let mut out = vec![0u8; row_bytes_slice * n_rows];
    for r in 0..n_rows {
        let src = &t.bytes[r * row_bytes_full..(r + 1) * row_bytes_full];
        let dst = &mut out[r * row_bytes_slice..(r + 1) * row_bytes_slice];
        for b in 0..col_len / block {
            let src_off = (cols.start as usize / block + b) * bb;
            dst[b * bb..(b + 1) * bb].copy_from_slice(&src[src_off..src_off + bb]);
        }
    }
    Ok(out)
}

/// The per-node slice of every weight tensor, as (name, kind, bytes) tuples.
/// Row-sharded (OutSharded) tensors are byte ranges; column-sharded (InSharded)
/// tensors are repacked.
pub fn node_tensor_slices(
    meta: &ModelMeta,
    weights: &Weights<'_>,
    node: &NodeShard,
) -> Result<Vec<(String, QuantKind, Vec<u8>)>, String> {
    let dim = meta.dim as usize;
    let q_dim = meta.q_dim() as usize;
    let kv_dim = meta.kv_dim() as usize;
    let ffn_dim = meta.ffn_dim() as usize;
    let mut out: Vec<(String, QuantKind, Vec<u8>)> = Vec::new();

    // embedding: vocab-row slice
    let embd = weights.get("token_embd")?;
    let rb = embd.kind.row_bytes(dim) as usize;
    out.push((
        "token_embd".into(),
        embd.kind,
        embd.bytes[node.embed_rows.start as usize * rb..node.embed_rows.end as usize * rb].to_vec(),
    ));

    for l in 0..meta.n_layers {
        // q/k/v: OutSharded rows
        for (name, slice, k) in [
            ("attn_q", &node.q_rows, dim),
            ("attn_k", &node.kv_rows, dim),
            ("attn_v", &node.kv_rows, dim),
        ] {
            let t = weights.get(&format!("blk.{l}.{name}"))?;
            let rb = t.kind.row_bytes(k) as usize;
            out.push((
                format!("blk.{l}.{name}"),
                t.kind,
                t.bytes[slice.start as usize * rb..slice.end as usize * rb].to_vec(),
            ));
        }

        // wo: InSharded cols (attn_output weight is [dim rows][q_dim cols])
        let wo = weights.get(&format!("blk.{l}.attn_output"))?;
        out.push((
            format!("blk.{l}.attn_output"),
            wo.kind,
            repack_cols(&wo, &node.wo_cols, q_dim, dim)?,
        ));

        // ffn_gate/ffn_up: OutSharded rows
        for name in ["ffn_gate", "ffn_up"] {
            let t = weights.get(&format!("blk.{l}.{name}"))?;
            let rb = t.kind.row_bytes(dim) as usize;
            out.push((
                format!("blk.{l}.{name}"),
                t.kind,
                t.bytes[node.ffn_rows.start as usize * rb..node.ffn_rows.end as usize * rb].to_vec(),
            ));
        }

        // ffn_down: InSharded cols ([dim rows][ffn cols])
        let w2 = weights.get(&format!("blk.{l}.ffn_down"))?;
        out.push((
            format!("blk.{l}.ffn_down"),
            w2.kind,
            repack_cols(&w2, &node.ffn_cols, ffn_dim, dim)?,
        ));

        // norms: replicated
        for n in ["attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"] {
            if let Ok(t) = weights.get(&format!("blk.{l}.{n}")) {
                out.push((format!("blk.{l}.{n}"), t.kind, t.bytes.to_vec()));
            }
        }

        if meta.is_moe() {
            let n_experts = meta.n_experts as usize;
            let moe_ffn = meta.moe_hidden_dim as usize;
            let local_ffn = (node.ffn_rows.end - node.ffn_rows.start) as usize;

            // gate router: replicated (all nodes compute the same logits)
            if let Ok(g) = weights.get(&format!("blk.{l}.ffn_gate_inp")) {
                out.push((format!("blk.{l}.ffn_gate_inp"), g.kind, g.bytes.to_vec()));
            }

            // w1/w3 (ffn_gate_exps / ffn_up_exps): OutSharded rows per expert
            // GGUF layout: [n_experts][moe_ffn][dim] — each expert's block is moe_ffn*rb(dim) bytes
            for (tensor_name, weight_name) in [("ffn_gate_exps", "ffn_gate_exps"), ("ffn_up_exps", "ffn_up_exps")] {
                let t = weights.get(&format!("blk.{l}.{weight_name}"))?;
                let rb = t.kind.row_bytes(dim) as usize;
                let expert_full = moe_ffn * rb;
                let row_off = node.ffn_rows.start as usize * rb;
                let row_len = local_ffn * rb;
                let mut sliced = Vec::with_capacity(n_experts * row_len);
                for e in 0..n_experts {
                    let base = e * expert_full;
                    if base + row_off + row_len > t.bytes.len() {
                        return Err(format!("expert tensor {} too small", tensor_name));
                    }
                    sliced.extend_from_slice(&t.bytes[base + row_off..base + row_off + row_len]);
                }
                out.push((format!("blk.{l}.{tensor_name}"), t.kind, sliced));
            }

            // w2 (ffn_down_exps): InSharded cols per expert (repack)
            // GGUF layout: [n_experts][dim][moe_ffn] — each expert's block is dim*rb(moe_ffn) bytes
            let w2 = weights.get(&format!("blk.{l}.ffn_down_exps"))?;
            let expert_w2_full = dim * w2.kind.row_bytes(moe_ffn) as usize;
            let local_rb = w2.kind.row_bytes(local_ffn) as usize;
            let mut w2_sliced = Vec::with_capacity(n_experts * dim * local_rb);
            for e in 0..n_experts {
                let base = e * expert_w2_full;
                if base + expert_w2_full > w2.bytes.len() {
                    return Err("expert w2 tensor too small".into());
                }
                let expert_tensor = Tensor { kind: w2.kind, bytes: &w2.bytes[base..base + expert_w2_full] };
                let repacked = repack_cols(&expert_tensor, &node.ffn_cols, moe_ffn, dim)?;
                w2_sliced.extend_from_slice(&repacked);
            }
            out.push((format!("blk.{l}.ffn_down_exps"), w2.kind, w2_sliced));
        }
    }
    out.push(("output_norm".into(), weights.get("output_norm")?.kind, weights.get("output_norm")?.bytes.to_vec()));

    // head: OutSharded vocab rows
    let head = weights.get("output")?;
    let rb = head.kind.row_bytes(dim) as usize;
    out.push((
        "output".into(),
        head.kind,
        head.bytes[node.cls_rows.start as usize * rb..node.cls_rows.end as usize * rb].to_vec(),
    ));
    Ok(out)
}

// ---------------------------------------------------------------------------
// worker: serve loop
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// node discovery (G3.2): UDP probe/reply so roots don't need --workers lists.
//
// Workers answer probes on UDP port DISCOVERY_PORT with their TCP listen
// port. The root broadcasts a probe, collects replies until the timeout,
// and assembles the node list. Wire format (ASCII):
//   probe:  "DLLAMA_DISCOVER?"
//   reply:  "DLLAMA_NODE <tcp_port>"
// ---------------------------------------------------------------------------

/// UDP port every worker listens on for discovery probes.
pub const DISCOVERY_PORT: u16 = 9990;
const PROBE: &[u8] = b"DLLAMA_DISCOVER?";

/// Spawn the discovery responder for a worker (runs until process exit).
/// Best effort: if the UDP port is taken (another worker on the same host),
/// the responder silently stays quiet — explicit addresses still work.
/// The reply carries a per-process instance id so roots dedupe a worker
/// that answers on several interfaces (loopback + LAN).
pub fn spawn_discovery_responder(tcp_port: u16) {
    std::thread::spawn(move || {
        let Ok(sock) = std::net::UdpSocket::bind(("0.0.0.0", DISCOVERY_PORT)) else {
            return; // another local worker already answers probes
        };
        let id = instance_id();
        let reply = format!("DLLAMA_NODE {tcp_port} {id:016x}").into_bytes();
        let mut buf = [0u8; 64];
        loop {
            match sock.recv_from(&mut buf) {
                Ok((n, peer)) => {
                    if &buf[..n] == PROBE {
                        let _ = sock.send_to(&reply, peer);
                    }
                }
                Err(_) => return,
            }
        }
    });
}

fn instance_id() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    t ^ (std::process::id() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Broadcast a discovery probe and collect node addresses until `timeout_ms`
/// elapses with no new replies. Probes go to the subnet broadcast and to
/// localhost (WSL2 NAT does not forward broadcast, but same-host roots find
/// their workers on 127.0.0.1). Returns `host:port` strings, deduplicated.
pub fn discover_nodes(timeout_ms: u64) -> Result<Vec<String>, String> {
    let sock = std::net::UdpSocket::bind(("0.0.0.0", 0)).map_err(|e| e.to_string())?;
    sock.set_broadcast(true).map_err(|e| e.to_string())?;
    for target in ["255.255.255.255", "127.0.0.1"] {
        let _ = sock.send_to(PROBE, (target, DISCOVERY_PORT));
    }
    // generous first window, then a shorter quiescence window
    let total = std::time::Duration::from_millis(timeout_ms);
    let quiesce = std::time::Duration::from_millis(200.min(timeout_ms));
    let start = std::time::Instant::now();
    // instance id -> (is_loopback, host:port): one entry per worker; a
    // worker answering on several interfaces keeps its non-loopback address
    let mut found: std::collections::HashMap<u64, (bool, String)> =
        std::collections::HashMap::new();
    loop {
        let elapsed = start.elapsed();
        let found_any = !found.is_empty();
        let remaining = if !found_any {
            total - elapsed
        } else {
            quiesce.min(total.saturating_sub(elapsed))
        };
        if remaining.is_zero() {
            break;
        }
        sock.set_read_timeout(Some(remaining)).map_err(|e| e.to_string())?;
        let mut buf = [0u8; 64];
        match sock.recv_from(&mut buf) {
            Ok((n, peer)) => {
                let msg = String::from_utf8_lossy(&buf[..n]);
                if let Some(rest) = msg.strip_prefix("DLLAMA_NODE ") {
                    let mut it = rest.trim().split_whitespace();
                    let (Some(port_s), Some(id_s)) = (it.next(), it.next()) else {
                        continue;
                    };
                    let (Ok(tcp_port), Ok(id)) = (port_s.parse::<u16>(), u64::from_str_radix(id_s, 16))
                    else {
                        continue;
                    };
                    let ip = peer.ip();
                    let is_loop = ip.is_loopback();
                    let addr = format!("{ip}:{tcp_port}");
                    found
                        .entry(id)
                        .and_modify(|(was_loop, prev)| {
                            // prefer the loopback address: a same-machine root
                            // reaches it regardless of firewall, and remote
                            // workers only ever answer from their LAN IP
                            if !*was_loop && is_loop {
                                *was_loop = is_loop;
                                *prev = addr.clone();
                            }
                        })
                        .or_insert((is_loop, addr));
                }
            }
            Err(_) => {
                if found.is_empty() && start.elapsed() < total {
                    if start.elapsed() < total / 2 {
                        for target in ["255.255.255.255", "127.0.0.1"] {
                            let _ = sock.send_to(PROBE, (target, DISCOVERY_PORT));
                        }
                    }
                    continue;
                }
                break;
            }
        }
    }
    let mut out: Vec<String> = found.into_values().map(|(_, a)| a).collect();
    out.sort();
    Ok(out)
}

pub fn serve_worker(host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port)).map_err(|e| format!("bind {host}:{port}: {e}"))?;
    spawn_discovery_responder(port);
    println!("🎧 worker listening on {host}:{port}");
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept error: {e}");
                continue;
            }
        };
        if let Err(e) = worker_session(&mut stream) {
            eprintln!("🚨 worker session ended: {e}");
        }
    }
    Ok(())
}

fn worker_session(stream: &mut TcpStream) -> Result<(), String> {
    // handshake
    let mut magic = [0u8; 4];
    stream.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if magic != MAGIC {
        return Err(format!("bad magic {magic:?}"));
    }
    let version = stream.r_u32()?;
    if version != PROTOCOL_VERSION {
        return Err(format!("protocol version mismatch: {version} != {PROTOCOL_VERSION}"));
    }
    let meta = read_meta(stream)?;
    let node_index = stream.r_u32()?;
    let n_nodes = stream.r_u32()?;

    // read tensors
    let n_tensors = stream.r_u32()?;
    let mut arena: Vec<Vec<u8>> = Vec::with_capacity(n_tensors as usize);
    let mut specs: Vec<(String, QuantKind, usize)> = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = stream.r_str()?;
        let kind = match stream.read_u8()? {
            0 => QuantKind::F32,
            1 => QuantKind::F16,
            2 => QuantKind::DllamaQ40,
            3 => QuantKind::GgufQ4_0,
            4 => QuantKind::GgufQ8_0,
            5 => QuantKind::GgufQ4K,
            6 => QuantKind::GgufQ6K,
            other => return Err(format!("unknown quant kind {other}")),
        };
        let bytes = stream.r_blob()?;
        specs.push((name, kind, arena.len()));
        arena.push(bytes);
    }

    // build the Weights map (borrow into `arena` — stable because arena is fully built)
    let mut map: HashMap<String, Tensor<'_>> = HashMap::with_capacity(specs.len());
    for (name, kind, idx) in &specs {
        map.insert(
            name.clone(),
            Tensor { kind: *kind, bytes: &arena[*idx] },
        );
    }
    let weights = Weights::from_map(map);

    // build the shard-aware executor
    let shard = shard_node(&meta, n_nodes, node_index).map_err(|e| e.to_string())?;
    println!("💡 worker {node_index}: {} tensors loaded, executing", specs.len());
    let mut inf = dllama_exec::Inference::new_distributed(&meta, &weights, Some(shard), node_index == 0)?;

    // ack
    stream.w_u32(ACK)?;

    // forward loop
    loop {
        let mut pkt = [0u8; 12];
        stream.read_exact(&mut pkt).map_err(|e| e.to_string())?;
        let cp = ControlPacket::decode(&pkt).ok_or("short control packet")?;
        if cp.batch_size == 0 {
            println!("👋 worker: stop packet received");
            return Ok(());
        }
        let mut sctx = dllama_exec::distributed::SyncCtx::Worker { stream }; inf.forward_with_stream(cp.token, cp.position, &mut sctx)?;
    }
}

trait ReadU8 {
    fn read_u8(&mut self) -> Result<u8, String>;
}

impl ReadU8 for TcpStream {
    fn read_u8(&mut self) -> Result<u8, String> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(b[0])
    }
}

#[cfg(test)]
mod discovery_tests {
    use super::*;

    /// Full probe/reply cycle over localhost (requires the responder socket;
    /// skipped if another test already holds the discovery port).
    #[test]
    fn discover_finds_local_worker() {
        // fake worker TCP port
        let fake_tcp_port = 19991u16;
        spawn_discovery_responder(fake_tcp_port);
        // give the responder a moment to bind
        std::thread::sleep(std::time::Duration::from_millis(100));
        let nodes = discover_nodes(1500).unwrap();
        assert_eq!(nodes.len(), 1, "dedup: one worker, one entry — got {nodes:?}");
        assert!(
            nodes[0].ends_with(&format!(":{fake_tcp_port}")),
            "discovery result: {nodes:?}"
        );
    }
}
