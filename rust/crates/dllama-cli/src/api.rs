//! G4.5: OpenAI-compatible API server with SSE streaming.
//!
//! Endpoints: GET /v1/models | POST /v1/chat/completions

use axum::{extract::State, response::sse::{Event, Sse}, routing::{get, post}, Json, Router};
use axum::response::IntoResponse;
use dllama_ir::ModelMeta;
use dllama_tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use futures::StreamExt;

// ---------------------------------------------------------------------------
// types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub seed: u64,
}

fn default_max_tokens() -> usize { 512 }
fn default_temperature() -> f32 { 0.0 }
fn default_top_p() -> f32 { 1.0 }

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
}

#[derive(Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ChoiceDelta {
    text: String,
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

pub struct ApiState {
    pub meta: ModelMeta,
    pub tokenizer: Tokenizer,
    // blocking inference behind a mutex (single request at a time)
    pub inference: Mutex<dllama_exec::Inference<'static>>,
    pub weights: &'static dllama_model::Weights<'static>,
}

// ---------------------------------------------------------------------------
// handlers
// ---------------------------------------------------------------------------

async fn list_models(State(s): State<Arc<ApiState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelInfo { id: s.meta.name.clone(), object: "model".into() }],
    })
}

async fn chat_completions(
    State(s): State<Arc<ApiState>>,
    Json(req): Json<ChatRequest>,
) -> axum::response::Response {
    // prompt via the GGUF chat template -> arch table -> ChatML chain
    let msgs: Vec<crate::chat_template::ChatMsg> = req
        .messages
        .iter()
        .map(|m| crate::chat_template::ChatMsg {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    let (prompt, src) = crate::chat_template::render_chat_prompt(
        s.tokenizer.chat_template.as_deref(),
        s.meta.arch,
        &msgs,
    );
    eprintln!("[api] prompt via {src}");

    let tokens = match s.tokenizer.encode(&prompt, true, true) {
        Ok(t) => t,
        Err(e) => return axum::http::StatusCode::BAD_REQUEST.into_response(),
    };
    let n = tokens.len();
    let seq_len = s.meta.seq_len as usize;
    let max_gen = req.max_tokens.min(seq_len.saturating_sub(n + 1));

    if req.stream {
        // SSE streaming via channel
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(100);
        let s2 = Arc::clone(&s);

        let mut sampler = crate::sampler::Sampler::new(
            req.temperature, req.top_k, req.top_p, req.seed, false,
        );
        tokio::task::spawn_blocking(move || {
            let _ = run_generation(s2, tokens, max_gen, &mut sampler, tx);
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(|text| {
            Ok::<_, Infallible>(Event::default().data(
                serde_json::json!({"object": "chat.completion.chunk", "choices": [{"text": text}]}).to_string(),
            ))
        });
        Sse::new(stream).into_response()
    } else {
        // non-streaming: collect all output
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(1000);
        let s2 = Arc::clone(&s);
        let mut sampler = crate::sampler::Sampler::new(
            req.temperature, req.top_k, req.top_p, req.seed, false,
        );
        tokio::task::spawn_blocking(move || { let _ = run_generation(s2, tokens, max_gen, &mut sampler, tx); });
        let mut output = String::new();
        while let Some(text) = rx.recv().await { // this runs in async context — need spawn_blocking
            output.push_str(&text);
        }
        Json(serde_json::json!({
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": output}}],
        })).into_response()
    }
}

fn run_generation(
    s: Arc<ApiState>,
    tokens: Vec<i32>,
    max_gen: usize,
    sampler: &mut crate::sampler::Sampler,
    tx: tokio::sync::mpsc::Sender<String>,
) -> Result<(), String> {
    let n = tokens.len();
    if n < 2 { return Err("prompt too short".into()); }
    // feed all but the last prompt token — batched when prefill is on
    let mut inf = s.inference.lock().map_err(|_| "mutex poisoned")?;
    if dllama_exec::prefill::prefill_enabled() {
        inf.prefill(&tokens[..n - 1], 0)?;
    } else {
        for i in 0..n - 1 {
            inf.forward(tokens[i] as u32, i as u32)?;
        }
    }
    let mut token = tokens[n - 1];
    let mut state = dllama_tokenizer::DecodeState::default();
    let mut pos = n - 1;
    for _ in 0..max_gen {
        let logits = inf.forward(token as u32, pos as u32)?;
        token = sampler.next(logits);
        if s.tokenizer.is_eos(token) { break; }
        if let Some(piece) = s.tokenizer.decode(&mut state, token) {
            let _ = tx.blocking_send(piece);
        }
        pos += 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// serve
// ---------------------------------------------------------------------------

pub async fn serve_api(
    meta: ModelMeta,
    tokenizer: Tokenizer,
    weights: &'static dllama_model::Weights<'static>,
    host: &str,
    port: u16,
) -> Result<(), String> {
    let inference = dllama_exec::Inference::new(&meta, weights)
        .map_err(|e| format!("inference init: {e}"))?;
    let state = Arc::new(ApiState {
        meta,
        tokenizer,
        inference: Mutex::new(inference),
        weights,
    });
    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);
    let addr = format!("{host}:{port}");
    println!("🚀 API server: http://{addr}/v1/");
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| e.to_string())?;
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}
