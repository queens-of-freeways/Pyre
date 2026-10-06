//! dllama-rs friendly CLI (G6.1): model auto-discovery + streaming chat.
//!
//! Users never type paths: models are discovered in `$DLLAMA_MODELS_DIR`,
//! `./models`, and the repo layout `<exe>/../../models`; commands take a
//! fuzzy name (`dllama-rs chat qwen3`), and with exactly one model present
//! even the name is optional.
//!
//! Multi-turn chat keeps a *token* history (each turn encoded as its own
//! chunk and appended to the persistent KV cache) instead of re-encoding the
//! whole conversation. The seams between chunks are ChatML special tokens,
//! which never merge, so the cached prefix stays consistent turn after turn.

use crate::g1;
use dllama_tokenizer::DecodeState;
use std::io::Write;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// model discovery
// ---------------------------------------------------------------------------

pub struct FoundModel {
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
}

fn model_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("DLLAMA_MODELS_DIR") {
        dirs.push(PathBuf::from(d));
    }
    dirs.push(PathBuf::from("models"));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            dirs.push(dir.join("..").join("models"));
            dirs.push(dir.join("..").join("..").join("models"));
        }
    }
    dirs
}

fn discover() -> Vec<FoundModel> {
    let mut found: Vec<FoundModel> = Vec::new();
    for dir in model_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext != "m" && ext != "gguf" {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&path) else { continue };
            if meta.is_dir() {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string();
            if found.iter().any(|m| m.path == path) {
                continue;
            }
            found.push(FoundModel {
                name,
                path,
                bytes: meta.len(),
            });
        }
    }
    // deterministic + friendliest (shortest) names first
    found.sort_by(|a, b| a.name.len().cmp(&b.name.len()).then(a.name.cmp(&b.name)));
    found
}

/// Resolve a user-supplied model argument: exact path â†’ itself; otherwise a
/// unique substring match on discovered models. `None` + exactly one model
/// â†’ that one.
fn resolve_model(arg: Option<&str>) -> Result<FoundModel, String> {
    let mut models = discover();
    let list = || {
        models
            .iter()
            .map(|m| format!("  {}", m.name))
            .collect::<Vec<_>>()
            .join("\n")
    };
    match arg {
        None => {
            if models.len() == 1 {
                Ok(models.remove(0))
            } else {
                Err(format!(
                    "no model given and {} models found â€” pick one:\n{}",
                    models.len(),
                    list()
                ))
            }
        }
        Some(a) => {
            let p = Path::new(a);
            if p.exists() {
                let name = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?")
                    .to_string();
                let bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                return Ok(FoundModel {
                    name,
                    path: p.to_path_buf(),
                    bytes,
                });
            }
            let lower = a.to_ascii_lowercase();
            let hits: Vec<usize> = models
                .iter()
                .enumerate()
                .filter(|(_, m)| m.name.to_ascii_lowercase().contains(&lower))
                .map(|(i, _)| i)
                .collect();
            match hits.len() {
                0 => Err(format!("no model matching '{a}' â€” available:\n{}", list())),
                1 => Ok(models.remove(hits[0])),
                _ => Err(format!(
                    "'{a}' is ambiguous:\n{}",
                    hits.iter()
                        .map(|&i| format!("  {}", models[i].name))
                        .collect::<Vec<_>>()
                        .join("\n")
                )),
            }
        }
    }
}

/// Find the `.t` tokenizer next to a dllama `.m` model (single .t wins;
/// otherwise the `dllama_model_X` â†’ `dllama_tokenizer_X` naming guess).
pub fn find_tokenizer(model: &Path) -> Result<Option<PathBuf>, String> {
    let Some(dir) = model.parent() else { return Ok(None) };
    let mut ts: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("t") {
                ts.push(p);
            }
        }
    }
    if ts.len() == 1 {
        return Ok(Some(ts.remove(0)));
    }
    let stem = model.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let guess = dir.join(format!("{}.t", stem.replace("model", "tokenizer")));
    if guess.exists() {
        return Ok(Some(guess));
    }
    if ts.is_empty() {
        Ok(None)
    } else {
        Err("multiple .t tokenizers next to the model â€” pass --tokenizer <path>".into())
    }
}

/// Split args into (positionals, flags-with-values kept adjacent).
fn split_args(args: &[String], value_flags: &[&str]) -> (Vec<String>, Vec<String>) {
    let mut pos = Vec::new();
    let mut flags = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with('-') && a.len() > 1 {
            flags.push(a.clone());
            if value_flags.contains(&a.as_str()) && i + 1 < args.len() {
                flags.push(args[i + 1].clone());
                i += 1;
            }
        } else {
            pos.push(a.clone());
        }
        i += 1;
    }
    (pos, flags)
}

// ---------------------------------------------------------------------------
// commands: models
// ---------------------------------------------------------------------------

pub fn cmd_models() -> i32 {
    let models = discover();
    if models.is_empty() {
        eprintln!("no models found â€” place .gguf/.m files in ./models or set DLLAMA_MODELS_DIR");
        return 1;
    }
    println!("Models found ({}):", models.len());
    for m in &models {
        println!(
            "  {:<44} {:>7.2} GB  {}",
            m.name,
            m.bytes as f64 / 1e9,
            m.path.display()
        );
    }
    println!("\nusage: dllama-rs chat <name>   (run `dllama-rs help` for more)");
    0
}

const VALUE_FLAGS: &[&str] = &[
    "--threads",
    "--max-seq-len",
    "--tokenizer",
    "--system",
    "--steps",
    "--prompt",
    "--model",
    "--temperature",
    "--top-k",
    "--top-p",
    "--seed",
];

/// Flags whose following token is a value (serve command).
pub const SERVE_FLAGS: &[&str] = &["--host", "--port", "--model", "--threads", "--max-seq-len"];

/// `--model` flag wins; otherwise the first positional (fuzzy); otherwise the
/// single discovered model.
pub fn resolve_from_args(args: &[String], value_flags: &[&str]) -> Result<FoundModel, String> {
    if let Some(p) = g1::flag(args, "--model") {
        return resolve_model(Some(p));
    }
    let (pos, _) = split_args(args, value_flags);
    resolve_model(pos.first().map(|s| s.as_str()))
}

// ---------------------------------------------------------------------------
// generation core (G6.2): full-template render + prefix-diff over the KV
// cache. Each turn renders the WHOLE conversation through the chat template
// (GGUF jinja -> arch table -> ChatML) and feeds only the tokens that extend
// what is already cached. Turns are bounded by special tokens, so the
// prefix is normally stable; if a tokenizer boundary ever shifts, the
// session restarts (fresh caches) and refeeds — correctness never depends
// on the fast path.
// ---------------------------------------------------------------------------

struct Loaded {
    inf: dllama_exec::Inference<'static>,
    tokenizer: dllama_tokenizer::Tokenizer,
    meta: dllama_ir::ModelMeta,
    weights: &'static dllama_model::Weights<'static>,
    seq_len: usize,
}

impl Loaded {
    /// Fresh KV caches (divergence restart). Weights, tokenizer and meta are
    /// process-lifetime, so this only reallocates cache memory.
    fn restart(&mut self) -> Result<(), String> {
        self.inf = dllama_exec::Inference::new(&self.meta, self.weights)?;
        Ok(())
    }
}

fn load_for_chat(path: &Path, max_seq_len: u32) -> Result<Loaded, String> {
    let source = g1::open_model(&path.to_string_lossy(), max_seq_len)?;
    let mut meta = source.meta(max_seq_len)?;
    let tokenizer_path = find_tokenizer(path)?;
    // build the tokenizer while `source` is still owned, then leak the model
    // container + weights onto the heap so the Inference's references stay
    // valid for the process lifetime (a stack `&weights` would be clobbered
    // once this frame returns — that bug class bit us once already).
    let tokenizer = source.tokenizer(
        tokenizer_path
            .as_deref()
            .map(|p| p.to_string_lossy().into_owned())
            .as_deref(),
    )?;
    let source: &'static g1::ModelSource = Box::leak(Box::new(source));
    let weights: &'static dllama_model::Weights<'static> = Box::leak(Box::new(source.weights()?));
    meta.sync_type = dllama_ir::FloatType::Q80; // validated default (G1 semantics)
    let inf = dllama_exec::Inference::new(&meta, weights)?;
    let seq_len = meta.seq_len as usize;
    Ok(Loaded {
        inf,
        tokenizer,
        meta,
        weights,
        seq_len,
    })
}

/// Tracks exactly what is in the KV cache: every token fed so far.
struct Session {
    fed: Vec<i32>,
    state: DecodeState,
    seq_len: usize,
}

impl Session {
    fn new(seq_len: usize) -> Self {
        Self {
            fed: Vec::new(),
            state: DecodeState::default(),
            seq_len,
        }
    }

    fn reset(&mut self) {
        self.fed.clear();
        self.state = DecodeState::default();
    }
}

/// Render the conversation, feed the new part of the prompt, generate up to
/// `steps` tokens (streaming to stdout), return the reply text.
fn generate_turn(
    loaded: &mut Loaded,
    session: &mut Session,
    messages: &[crate::chat_template::ChatMsg],
    sampler: &mut crate::sampler::Sampler,
    steps: usize,
) -> Result<String, String> {
    let (prompt, src) = crate::chat_template::render_chat_prompt(
        loaded.tokenizer.chat_template.as_deref(),
        loaded.meta.arch,
        messages,
    );
    if session.fed.is_empty() {
        eprintln!("[chat-template] prompt via {src}");
    }
    let tokens = loaded.tokenizer.encode(&prompt, true, true)?;
    let n = tokens.len();
    if n < 2 {
        return Err("prompt encoded to fewer than 2 tokens".into());
    }
    if n + 2 >= session.seq_len {
        return Err("context window exhausted".into());
    }
    // KV continuation: the new tokenization must extend what is cached.
    let common = session
        .fed
        .iter()
        .zip(&tokens)
        .take_while(|(a, b)| a == b)
        .count();
    if common < session.fed.len() || n - 1 < common {
        loaded.restart()?;
        session.reset();
    }
    let start = session.fed.len();
    let chunk: Vec<i32> = tokens[start..n - 1].to_vec();
    if !chunk.is_empty() {
        if dllama_exec::prefill::prefill_enabled() {
            loaded.inf.prefill(&chunk, start)?;
            session.fed.extend_from_slice(&chunk);
        } else {
            for &t in &chunk {
                loaded.inf.forward(t as u32, session.fed.len() as u32)?;
                session.fed.push(t);
            }
        }
    }
    // prompt end -> first generated token (sampled; not printed — it is the
    // reply's first token and comes from the prompt's last-token logits)
    let logits = loaded
        .inf
        .forward(tokens[n - 1] as u32, session.fed.len() as u32)?;
    session.fed.push(tokens[n - 1]);
    let mut token = sampler.next(logits);
    let mut reply = String::new();
    for _ in 0..steps {
        if loaded.tokenizer.is_eos(token) {
            break;
        }
        if let Some(piece) = loaded.tokenizer.decode(&mut session.state, token) {
            print!("{piece}");
            let _ = std::io::stdout().flush();
            reply.push_str(&piece);
        }
        if session.fed.len() + 1 >= session.seq_len {
            break;
        }
        let logits = loaded
            .inf
            .forward(token as u32, session.fed.len() as u32)?;
        session.fed.push(token);
        let next = sampler.next(logits);
        if loaded.tokenizer.is_eos(next) {
            break;
        }
        token = next;
    }
    Ok(reply)
}

fn default_system() -> String {
    "You are a helpful assistant.".into()
}

fn chat_messages(args: &[String], user: &str) -> Vec<crate::chat_template::ChatMsg> {
    let system = g1::flag(args, "--system")
        .map(|s| s.to_string())
        .unwrap_or_else(default_system);
    vec![
        crate::chat_template::ChatMsg {
            role: "system".into(),
            content: system,
        },
        crate::chat_template::ChatMsg {
            role: "user".into(),
            content: user.into(),
        },
    ]
}

/// Interactive multi-turn chat.
pub fn cmd_chat(args: &[String]) -> i32 {
    g1::apply_threads_flag(args);
    let (pos, _flags) = split_args(args, VALUE_FLAGS);
    let _ = &pos;
    let model = match resolve_from_args(args, VALUE_FLAGS) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let max_seq_len: u32 = g1::flag(args, "--max-seq-len")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut loaded = match load_for_chat(&model.path, max_seq_len) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("load failed: {e}");
            return 1;
        }
    };
    let mut sampler = crate::sampler::sampler_from_args(args);
    let steps = g1::flag(args, "--steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let mode = if sampler.is_greedy() {
        String::new()
    } else {
        format!(" (temp {}, top-k {}, top-p {})", sampler.temperature, sampler.top_k, sampler.top_p)
    };
    println!(
        "dllama-rs chat — {} ({:.2} GB){}, type a message (empty line or Ctrl-C to quit)",
        model.name,
        model.bytes as f64 / 1e9,
        mode
    );
    let mut session = Session::new(loaded.seq_len);
    let system = g1::flag(args, "--system")
        .map(|s| s.to_string())
        .unwrap_or_else(default_system);
    let mut messages = vec![crate::chat_template::ChatMsg {
        role: "system".into(),
        content: system,
    }];
    loop {
        print!("\nyou> ");
        let _ = std::io::stdout().flush();
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() || input.trim().is_empty() {
            println!("\nbye");
            return 0;
        }
        let input = input.trim().to_string();
        messages.push(crate::chat_template::ChatMsg {
            role: "user".into(),
            content: input,
        });
        print!("\nassistant> ");
        let _ = std::io::stdout().flush();
        match generate_turn(&mut loaded, &mut session, &messages, &mut sampler, steps) {
            Ok(reply) => {
                println!();
                messages.push(crate::chat_template::ChatMsg {
                    role: "assistant".into(),
                    content: reply,
                });
            }
            Err(e) => {
                eprintln!("\n{e}");
                return 1;
            }
        }
    }
}

/// One-shot generation: `dllama-rs ask [model] "question"`.
pub fn cmd_ask(args: &[String]) -> i32 {
    g1::apply_threads_flag(args);
    let (pos, _flags) = split_args(args, VALUE_FLAGS);
    let model = match resolve_from_args(args, VALUE_FLAGS) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let prompt_skip = if g1::flag(args, "--model").is_some() { 0 } else { 1 };
    let prompt = g1::flag(args, "--prompt")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            pos.iter()
                .skip(prompt_skip)
                .cloned()
                .collect::<Vec<_>>()
                .join(" ")
        });
    if prompt.trim().is_empty() {
        eprintln!(
            "ask: give a prompt, e.g.  dllama-rs ask {} \"what is rust?\"",
            model.name
        );
        return 1;
    }
    let max_seq_len: u32 = g1::flag(args, "--max-seq-len")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut loaded = match load_for_chat(&model.path, max_seq_len) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("load failed: {e}");
            return 1;
        }
    };
    let mut sampler = crate::sampler::sampler_from_args(args);
    let steps = g1::flag(args, "--steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let messages = chat_messages(args, &prompt);
    let mut session = Session::new(loaded.seq_len);
    match generate_turn(&mut loaded, &mut session, &messages, &mut sampler, steps) {
        Ok(_) => {
            println!();
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

// ---------------------------------------------------------------------------
// commands: ppl / serve bridges (reuse the advanced commands)
// ---------------------------------------------------------------------------

/// `dllama-rs ppl <model> [--prompt "text"]` â†’ the perplexity command with
/// the model path injected.
pub fn cmd_ppl(args: &[String]) -> i32 {
    let (pos, mut flags) = split_args(args, VALUE_FLAGS);
    let model = match resolve_from_args(args, VALUE_FLAGS) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let mut full: Vec<String> = vec!["--model".into(), model.path.to_string_lossy().into_owned()];
    match find_tokenizer(&model.path) {
        Ok(Some(t)) => {
            full.push("--tokenizer".into());
            full.push(t.to_string_lossy().into_owned());
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    }
    if !flags.iter().any(|f| f == "--prompt") && pos.len() >= 2 {
        full.push("--prompt".into());
        full.push(pos[1..].iter().cloned().collect::<Vec<_>>().join(" "));
    }
    full.append(&mut flags);
    g1::run_perplexity(&full)
}
