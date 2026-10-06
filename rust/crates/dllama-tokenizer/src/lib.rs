//! dllama-tokenizer: `.t` tokenizer format + BPE encode/decode.
//!
//! Faithful port of `src/tokenizer.cpp` (constructor, `encode`, `decode`):
//! - new format magic `0x567124`: i32 headerSize, KV header, optional chat
//!   template + eos ids, then per token: f32 score, i32 length, raw bytes.
//! - encode: greedy char-loop with special-token prefix matching + best-score
//!   pair merging (llama2.c style BPE), lowest-id candidate wins ties.

use std::collections::HashMap;

const MAGIC_OLD: i32 = 0x567123;
const MAGIC_NEW: i32 = 0x567124;

/// GPT-2 byte encoder maps bytes to printable Unicode codepoints.
/// This builds the reverse map: codepoint → original byte.
/// See https://github.com/openai/gpt-2/blob/master/src/encoder.py
fn build_gpt2_byte_decode_map() -> HashMap<char, u8> {
    let mut map = HashMap::new();
    // printable bytes map to themselves
    for b in 33u32..=126 {
        map.insert(char::from_u32(b).unwrap(), b as u8);
    }
    for b in 161u32..=172 {
        map.insert(char::from_u32(b).unwrap(), b as u8);
    }
    for b in 174u32..=255 {
        map.insert(char::from_u32(b).unwrap(), b as u8);
    }
    // non-printable bytes map to codepoints starting at 256
    let mut n = 0u32;
    for b in 0u32..=255 {
        if !(33..=126).contains(&b) && !(161..=172).contains(&b) && !(174..=255).contains(&b) {
            map.insert(char::from_u32(256 + n).unwrap(), b as u8);
            n += 1;
        }
    }
    map
}

/// Decode a GPT-2 byte-encoded string to raw bytes.
/// Each Unicode codepoint in the string maps to exactly one byte.
pub fn gpt2_decode(s: &str) -> Vec<u8> {
    let map = build_gpt2_byte_decode_map();
    s.chars().filter_map(|c| map.get(&c).copied()).collect()
}

// Header keys — mirror `TokenizerHeaderKey` (tokenizer.hpp:23).
const TOK_VERSION: i32 = 0;
const TOK_VOCAB_SIZE: i32 = 1;
const MAX_TOKEN_LENGTH: i32 = 2;
const BOS_ID: i32 = 3;
const EOS_ID: i32 = 4;
const PAD_ID: i32 = 5;
const CHAT_EOS_ID: i32 = 6;
const CHAT_TEMPLATE: i32 = 7;
const CHAT_STOP: i32 = 8;
const N_EOS_TOKENS: i32 = 9;
const ADD_BOS: i32 = 10;

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn i32(&mut self) -> Result<i32, String> {
        if self.pos + 4 > self.data.len() {
            return Err("tokenizer file truncated (i32)".into());
        }
        let v = i32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }
    fn f32(&mut self) -> Result<f32, String> {
        if self.pos + 4 > self.data.len() {
            return Err("tokenizer file truncated (f32)".into());
        }
        let v = f32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err(format!("tokenizer file truncated (need {n} bytes)"));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn skip(&mut self, n: usize) -> Result<(), String> {
        if self.pos + n > self.data.len() {
            return Err("tokenizer file truncated (skip)".into());
        }
        self.pos += n;
        Ok(())
    }
}

pub struct Tokenizer {
    pub vocab: Vec<Vec<u8>>,
    pub scores: Vec<f32>,
    pub bos_id: i32,
    pub eos_token_ids: Vec<i32>,
    pub add_bos: bool,
    pub max_token_length: u32,
    pub chat_template: Option<String>,
    /// string -> candidate ids (ascending, first = lowest id, like the C++ FNV map)
    regular: HashMap<Vec<u8>, Vec<i32>>,
    /// special token ids, ascending (ids >= bos_id)
    special: Vec<i32>,
}

impl Tokenizer {
    pub fn load(path: &str) -> Result<Tokenizer, String> {
        let data = std::fs::read(path).map_err(|e| format!("cannot open tokenizer file ({path}): {e}"))?;
        Self::parse(&data)
    }

    /// Build a Tokenizer from raw GGUF metadata (tokens, scores, ids).
    /// Same encoding algorithm as the `.t` format — no pre-tokenization.
    /// Vocab entries are decoded from GPT-2 byte encoding to raw bytes.
    pub fn from_gguf_data(
        vocab: Vec<Vec<u8>>,
        scores: Vec<f32>,
        bos_id: i32,
        eos_ids: Vec<i32>,
        add_bos: bool,
        chat_template: Option<String>,
    ) -> Result<Tokenizer, String> {
        // decode vocab from GPT-2 byte encoding to raw bytes
        // (GGUF stores tokens with "Ġ" for space, "Ċ" for newline, etc.)
        let vocab: Vec<Vec<u8>> = vocab
            .iter()
            .map(|v| gpt2_decode(&String::from_utf8_lossy(v)))
            .collect();
        if vocab.is_empty() {
            return Err("empty GGUF vocab".into());
        }
        if bos_id < 0 || bos_id as usize >= vocab.len() {
            return Err(format!("invalid bos_id {bos_id} for vocab of {}", vocab.len()));
        }
        let max_token_length = vocab.iter().map(|v| v.len()).max().unwrap_or(0);
        if max_token_length < 1 {
            return Err("invalid max token length".into());
        }
        let regular_size = bos_id as usize;
        let mut regular: HashMap<Vec<u8>, Vec<i32>> = HashMap::new();
        for (id, s) in vocab.iter().enumerate().take(regular_size) {
            regular.entry(s.clone()).or_default().push(id as i32);
        }
        let special: Vec<i32> = (regular_size..vocab.len()).map(|i| i as i32).collect();
        Ok(Tokenizer {
            vocab,
            scores,
            bos_id,
            eos_token_ids: eos_ids,
            add_bos,
            max_token_length: max_token_length as u32,
            chat_template,
            regular,
            special,
        })
    }

    pub fn parse(data: &[u8]) -> Result<Tokenizer, String> {
        let mut cur = Cursor { data, pos: 0 };
        let magic = cur.i32()?;
        let mut vocab_size: i32 = 0;
        let mut max_token_length: i32 = 0;
        let mut bos_id: i32 = -1;
        let mut eos_ids: Vec<i32> = Vec::new();
        let mut add_bos = true;
        let mut chat_template_length: i32 = -1;
        let mut n_eos_tokens: i32 = 0;

        match magic {
            MAGIC_OLD => {
                // TokenizerOldHeader: vocabSize u32, maxTokenLength u32, bosId, eosId, padId
                vocab_size = cur.i32()?;
                max_token_length = cur.i32()?;
                bos_id = cur.i32()?;
                eos_ids.push(cur.i32()?);
                cur.i32()?; // padId (ignored)
            }
            MAGIC_NEW => {
                let header_size = cur.i32()? as usize;
                if header_size < 8 || header_size > data.len() + 8 {
                    return Err(format!("invalid tokenizer header size: {header_size}"));
                }
                let n_kv = (header_size - 8) / 4; // i32 count of KV data
                let kv_bytes = cur.bytes(n_kv * 4)?;
                let mut kv_pos = 0usize;
                while kv_pos + 8 <= kv_bytes.len() {
                    let key = i32::from_le_bytes(kv_bytes[kv_pos..kv_pos + 4].try_into().unwrap());
                    let value = i32::from_le_bytes(kv_bytes[kv_pos + 4..kv_pos + 8].try_into().unwrap());
                    match key {
                        TOK_VERSION => {
                            if value != 1 {
                                return Err("old tokenizer version, regenerate the tokenizer".into());
                            }
                        }
                        TOK_VOCAB_SIZE => vocab_size = value,
                        MAX_TOKEN_LENGTH => max_token_length = value,
                        BOS_ID => bos_id = value,
                        EOS_ID | CHAT_EOS_ID => eos_ids.push(value),
                        CHAT_TEMPLATE => chat_template_length = value,
                        CHAT_STOP => cur.skip(value.max(0) as usize)?, // fseek during header parse
                        PAD_ID => {}
                        N_EOS_TOKENS => n_eos_tokens = value,
                        ADD_BOS => add_bos = value == 1,
                        other => return Err(format!("invalid tokenizer header key: {other}")),
                    }
                    kv_pos += 8;
                }
            }
            other => return Err(format!("invalid tokenizer magic: {other:#x}")),
        }

        if max_token_length < 1 {
            return Err("invalid tokenizer max token length".into());
        }

        let chat_template = if chat_template_length > 0 {
            let b = cur.bytes(chat_template_length as usize)?;
            Some(String::from_utf8_lossy(b).into_owned())
        } else {
            None
        };
        if n_eos_tokens > 0 {
            for _ in 0..n_eos_tokens {
                eos_ids.push(cur.i32()?);
            }
        }

        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(vocab_size as usize);
        let mut scores: Vec<f32> = Vec::with_capacity(vocab_size as usize);
        for _ in 0..vocab_size {
            let score = cur.f32()?;
            let len = cur.i32()?;
            if len < 0 {
                return Err("invalid token length".into());
            }
            vocab.push(cur.bytes(len as usize)?.to_vec());
            scores.push(score);
        }

        // C++: regular = [0, bos_id), special = [bos_id, vocab_size) (tokenizer.cpp:161)
        let regular_size = bos_id.max(0) as usize;
        let mut regular: HashMap<Vec<u8>, Vec<i32>> = HashMap::new();
        for (id, s) in vocab.iter().enumerate().take(regular_size) {
            regular.entry(s.clone()).or_default().push(id as i32);
        }
        let special: Vec<i32> = (regular_size..vocab.len() as usize).map(|i| i as i32).collect();

        Ok(Tokenizer {
            vocab,
            scores,
            bos_id,
            eos_token_ids: eos_ids,
            add_bos,
            max_token_length: max_token_length as u32,
            chat_template,
            regular,
            special,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn find_regular(&self, piece: &[u8]) -> Option<i32> {
        self.regular.get(piece).and_then(|v| v.first().copied())
    }

    /// C++ findSpecialTokenStartWith: first (lowest-id) special token that is a
    /// prefix of `piece`.
    fn find_special_prefix(&self, piece: &[u8]) -> Option<i32> {
        for &id in &self.special {
            let s = &self.vocab[id as usize];
            if piece.starts_with(s.as_slice()) {
                return Some(id);
            }
        }
        None
    }

    pub fn is_eos(&self, token: i32) -> bool {
        self.eos_token_ids.contains(&token)
    }

    /// Port of `Tokenizer::encode` (tokenizer.cpp:255).
    pub fn encode(&self, text: &str, is_start: bool, add_special_tokens: bool) -> Result<Vec<i32>, String> {
        let mut tokens: Vec<i32> = Vec::new();
        if is_start && self.add_bos && self.bos_id >= 0 {
            tokens.push(self.bos_id);
        }
        let text = text.as_bytes();
        let mut buf: Vec<u8> = Vec::new();
        let mut i = 0usize;
        while i < text.len() {
            if add_special_tokens {
                if let Some(id) = self.find_special_prefix(&text[i..]) {
                    tokens.push(id);
                    i += self.vocab[id as usize].len();
                    continue;
                }
            }
            buf.push(text[i]);
            i += 1;
            if let Some(id) = self.find_regular(&buf) {
                tokens.push(id);
                buf.clear();
            }
        }
        if !buf.is_empty() {
            return Err("tokenizer error: unmatched trailing text".into());
        }

        // merge the best consecutive pair each iteration (best score wins, first on tie)
        loop {
            let mut best_score = -1e10f32;
            let mut best_id: i32 = -1;
            let mut best_idx: isize = -1;
            for i in 0..tokens.len().saturating_sub(1) {
                let t0 = tokens[i] as usize;
                let t1 = tokens[i + 1] as usize;
                let len0 = self.vocab[t0].len();
                let len1 = self.vocab[t1].len();
                if len0 + len1 > self.max_token_length as usize {
                    continue;
                }
                let mut cat = self.vocab[t0].clone();
                cat.extend_from_slice(&self.vocab[t1]);
                if let Some(id) = self.find_regular(&cat) {
                    if self.scores[id as usize] > best_score {
                        best_score = self.scores[id as usize];
                        best_id = id;
                        best_idx = i as isize;
                    }
                }
            }
            if best_idx == -1 {
                break;
            }
            let idx = best_idx as usize;
            tokens[idx] = best_id;
            tokens.remove(idx + 1);
        }
        Ok(tokens)
    }

    /// Decode one token: BOS -> None; EOS -> flush pending buffer (returns
    /// Some(text) if any); other -> append piece, return complete-utf8 delta.
    /// (Simplified port of `decode` + `detokUtf8`: returns the longest valid
    /// utf-8 prefix, retaining incomplete sequences — same observable behavior
    /// for valid text.)
    pub fn decode(&self, state: &mut DecodeState, token: i32) -> Option<String> {
        if token == self.bos_id {
            return None;
        }
        if self.is_eos(token) {
            let out = String::from_utf8_lossy(&state.buffer).into_owned();
            state.buffer.clear();
            return if out.is_empty() { None } else { Some(out) };
        }
        let piece = &self.vocab[token as usize];
        state.buffer.extend_from_slice(piece);
        // flush the longest valid utf-8 prefix
        let valid = match std::str::from_utf8(&state.buffer) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid > 0 {
            let out = String::from_utf8_lossy(&state.buffer[..valid]).into_owned();
            state.buffer.drain(..valid);
            Some(out)
        } else {
            None
        }
    }
}

/// Mutable decode buffer (C++ keeps this inside Tokenizer; we keep it outside
/// for a clean API).
#[derive(Default)]
pub struct DecodeState {
    buffer: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// minimal synthetic .t (new format): vocab "a","b","ab","<bos>","<eos>"
    fn synthetic_tokenizer() -> Vec<u8> {
        let vocab: &[(&str, f32)] = &[("a", -1.0), ("b", -2.0), ("ab", -0.5), ("<s>", 0.0), ("</s>", 0.0)];
        let kv: Vec<(i32, i32)> = vec![
            (TOK_VERSION, 1),
            (TOK_VOCAB_SIZE, vocab.len() as i32),
            (MAX_TOKEN_LENGTH, 4),
            (BOS_ID, 3),
            (N_EOS_TOKENS, 1),
            (ADD_BOS, 1),
        ];
        let header_size = 8 + kv.len() * 8;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&MAGIC_NEW.to_le_bytes());
        buf.extend_from_slice(&(header_size as i32).to_le_bytes());
        for (k, v) in &kv {
            buf.extend_from_slice(&k.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // N_EOS_TOKENS=1 in the header -> exactly one eos id follows the KV block
        buf.extend_from_slice(&(4i32).to_le_bytes()); // eos = 4 ("</s>")
        for (s, score) in vocab {
            buf.extend_from_slice(&score.to_le_bytes());
            let b = s.as_bytes();
            buf.extend_from_slice(&(b.len() as i32).to_le_bytes());
            buf.extend_from_slice(b);
        }
        buf
    }

    #[test]
    fn loads_and_encodes() {
        let data = synthetic_tokenizer();
        let t = Tokenizer::parse(&data).unwrap();
        assert_eq!(t.vocab_size(), 5);
        assert_eq!(t.bos_id, 3);
        assert_eq!(t.eos_token_ids, vec![4]);
        assert!(t.add_bos);

        // "ab" merges: chars a,b then best pair merge -> "ab" (score -0.5)
        let toks = t.encode("ab", false, false).unwrap();
        assert_eq!(toks, vec![2]);
        let toks = t.encode("aab", false, false).unwrap();
        assert_eq!(toks, vec![0, 2]);
        // bos + special-token matching
        let toks = t.encode("a</s>b", true, true).unwrap();
        assert_eq!(toks, vec![3, 0, 4, 1]);
    }

    #[test]
    fn decode_roundtrip() {
        let data = synthetic_tokenizer();
        let t = Tokenizer::parse(&data).unwrap();
        let mut st = DecodeState::default();
        assert_eq!(t.decode(&mut st, 3), None); // bos
        assert_eq!(t.decode(&mut st, 2), Some("ab".into()));
        assert_eq!(t.decode(&mut st, 4), None); // eos, empty buffer
        let mut st2 = DecodeState::default();
        assert_eq!(t.decode(&mut st2, 0), Some("a".into()));
        assert_eq!(t.decode(&mut st2, 1), Some("b".into()));
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(Tokenizer::parse(&[1, 2, 3, 4]).is_err());
    }
}
