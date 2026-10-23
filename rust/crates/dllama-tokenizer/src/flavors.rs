//! GGUF tokenizer flavors ported from llama.cpp `src/llama-vocab.cpp` (MIT,
//! ggml-org/llama.cpp). The original BPE-greedy path in `lib.rs` stays the
//! default (bit-stable); this module adds:
//!
//! - **SPM**  (model "llama"): sentencepiece — score-priority bigram merge
//!   over UTF-8 symbols + recursive resegmentation + byte fallback.
//! - **WPM**  (model "bert"): wordpiece — whitespace/punct word split,
//!   greedy longest match per word with "▁" prefix, all-or-UNK.
//!   (Minimal unicode: ASCII + common ranges; U+0300-036F combining marks
//!   stripped when accent-stripping is on — documented divergence from
//!   llama.cpp's full category tables.)
//! - **UGM**  (model "t5"): unigram — Viterbi lattice + precompiled
//!   charsmap normalization (XCDA double-array, ported verbatim).
//! - **RWKV** (model "rwkv"): escaped-vocab greedy longest match.
//!
//! GGUF token types (convert_hf_to_gguf.py): 1=NORMAL 2=UNKNOWN 3=CONTROL
//! 4=USER_DEFINED 5=UNUSED 6=BYTE.

use crate::Tokenizer;
use std::collections::{BinaryHeap, HashMap};
use unicode_normalization::UnicodeNormalization;

pub const TT_NORMAL: i32 = 1;
pub const TT_UNKNOWN: i32 = 2;
pub const TT_CONTROL: i32 = 3;
pub const TT_USER_DEFINED: i32 = 4;
pub const TT_UNUSED: i32 = 5;
pub const TT_BYTE: i32 = 6;

const ESCAPED_SPACE: &[u8] = b"\xe2\x96\x81"; // U+2581

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flavor {
    Bpe,
    Spm,
    Wpm,
    Ugm,
    Rwkv,
}

/// `tokenizer.ggml.model` -> flavor (unknown models fall back to the BPE
/// path — the historical behavior — with a warning).
pub fn flavor_from_model(model: &str) -> Flavor {
    match model {
        "llama" => Flavor::Spm,
        "bert" => Flavor::Wpm,
        "t5" => Flavor::Ugm,
        "rwkv" => Flavor::Rwkv,
        other => {
            eprintln!(
                "[tokenizer] ggml model '{other}' has no dedicated flavor; using the BPE path"
            );
            Flavor::Bpe
        }
    }
}

/// Everything the GGUF loader collected for a tokenizer build.
pub struct GgufTokenizerInput {
    pub model: String,
    pub vocab_raw: Vec<Vec<u8>>,
    pub scores: Vec<f32>,
    pub types: Vec<i32>,
    pub bos_id: i32,
    pub eos_ids: Vec<i32>,
    pub add_bos: bool,
    pub chat_template: Option<String>,
    /// explicit unk id from `tokenizer.ggml.unk_token_id` (-1 = auto-detect)
    pub unk_id: i32,
    pub charsmap: Vec<u8>,
    pub metaspace_add_space_prefix: bool,
    pub metaspace_escape: bool,
    pub remove_extra_ws: bool,
    pub wpm_lowercase: bool,
}

/// Build a Tokenizer for any flavor. BPE delegates to the historical
/// `Tokenizer::from_gguf_data` (bit-stable behavior) and just tags the flavor.
pub fn build_gguf_tokenizer(input: GgufTokenizerInput) -> Result<Tokenizer, String> {
    let flavor = flavor_from_model(&input.model);
    if flavor == Flavor::Bpe {
        let t = Tokenizer::from_gguf_data(
            input.vocab_raw,
            input.scores,
            input.bos_id,
            input.eos_ids,
            input.add_bos,
            input.chat_template,
        )?;
        return Ok(Tokenizer { flavor, ..t });
    }

    let n = input.vocab_raw.len();
    if n == 0 {
        return Err("empty GGUF vocab".into());
    }

    // RWKV vocab pieces are escape-encoded; unescape for matching + decode.
    let vocab: Vec<Vec<u8>> = match flavor {
        Flavor::Rwkv => input.vocab_raw.iter().map(|p| unescape_rwkv(p)).collect(),
        _ => input.vocab_raw,
    };

    let types = if input.types.len() == n {
        input.types.clone()
    } else if flavor == Flavor::Spm {
        // legacy SPM vocabs (no tokenizer.ggml.types): derive from the piece
        // text like llama.cpp's legacy path — "<0xXX>" -> BYTE (byte
        // fallback), "<unk>" -> UNKNOWN, sentencepiece controls -> CONTROL
        vocab.iter()
            .map(|p| {
                if p.len() == 6 && p.starts_with(b"<0x") && p.ends_with(b">") {
                    TT_BYTE
                } else if p == b"<unk>" {
                    TT_UNKNOWN
                } else if p == b"<s>" || p == b"</s>" || p == b"<pad>" {
                    TT_CONTROL
                } else {
                    TT_NORMAL
                }
            })
            .collect()
    } else {
        vec![TT_NORMAL; n]
    };

    // piece -> id map (last id wins, like llama.cpp's unordered_map insert)
    let mut piece_map: HashMap<Vec<u8>, i32> = HashMap::new();
    for (id, piece) in vocab.iter().enumerate() {
        piece_map.insert(piece.clone(), id as i32);
    }

    // specials (fragment-splitting + prompt parsing): CONTROL | USER_DEFINED
    let special: Vec<i32> = (0..n)
        .filter(|&i| types[i] == TT_CONTROL || types[i] == TT_USER_DEFINED)
        .map(|i| i as i32)
        .collect();

    // byte fallback map from BYTE-type "<0xXX>" pieces
    let mut byte_tokens = [-1i32; 256];
    for (id, piece) in vocab.iter().enumerate() {
        if types[id] == TT_BYTE
            && piece.len() == 6
            && piece.starts_with(b"<0x")
            && piece.ends_with(b">")
        {
            if let Ok(hex) = std::str::from_utf8(&piece[3..5]) {
                if let Ok(v) = u8::from_str_radix(hex, 16) {
                    byte_tokens[v as usize] = id as i32;
                }
            }
        }
    }

    let unk_id = if input.unk_id >= 0 && (input.unk_id as usize) < n {
        input.unk_id // explicit tokenizer.ggml.unk_token_id wins (llama.cpp)
    } else if let Some(i) = (0..n).find(|&i| vocab[i] == b"[UNK]") {
        i as i32 // WPM convention: the [UNK] piece itself
    } else {
        (0..n)
            .find(|&i| types[i] == TT_UNKNOWN)
            .map(|i| i as i32)
            .unwrap_or(0)
    };

    // UGM: min score over NORMAL tokens + charsmap parse + user-defined pieces
    let mut min_score = f32::MAX;
    for (id, s) in input.scores.iter().enumerate() {
        if id < n && types[id] == TT_NORMAL {
            min_score = min_score.min(*s);
        }
    }
    if min_score == f32::MAX {
        min_score = 0.0;
    }
    let (ugm_xcda, ugm_replacements) = parse_charsmap(&input.charsmap)?;
    let ugm_user_defined: Vec<Vec<u8>> = (0..n)
        .filter(|&i| types[i] == TT_USER_DEFINED)
        .map(|i| vocab[i].clone())
        .collect();

    let max_token_length = vocab.iter().map(|v| v.len()).max().unwrap_or(1);
    if max_token_length < 1 {
        return Err("invalid max token length".into());
    }

    Ok(Tokenizer {
        vocab,
        scores: input.scores,
        bos_id: input.bos_id,
        eos_token_ids: input.eos_ids,
        add_bos: input.add_bos,
        max_token_length: max_token_length as u32,
        chat_template: input.chat_template,
        flavor,
        piece_map,
        token_types: types,
        byte_tokens,
        unk_id,
        ugm_xcda,
        ugm_replacements,
        ugm_user_defined,
        min_score,
        add_space_prefix: input.metaspace_add_space_prefix,
        escape_ws: input.metaspace_escape,
        remove_extra_ws: input.remove_extra_ws,
        wpm_lowercase: input.wpm_lowercase,
        regular: HashMap::new(),
        special,
    })
}

// ---------------------------------------------------------------------------
// shared: special-token fragment splitting (llama.cpp tokenizer_st_partition)
// ---------------------------------------------------------------------------

enum Frag<'a> {
    Raw(&'a [u8]),
    Tok(i32),
}

fn split_fragments<'a>(t: &Tokenizer, text: &'a [u8], parse_special: bool) -> Vec<Frag<'a>> {
    let mut frags = Vec::new();
    let mut raw_start = 0usize;
    let mut i = 0usize;
    while i < text.len() {
        if parse_special {
            // longest special-token match at i
            let mut best: Option<(usize, i32)> = None;
            for &id in &t.special {
                let piece = &t.vocab[id as usize];
                if text[i..].starts_with(piece)
                    && best.map(|(l, _)| piece.len() > l).unwrap_or(true)
                {
                    best = Some((piece.len(), id));
                }
            }
            if let Some((len, id)) = best {
                if i > raw_start {
                    frags.push(Frag::Raw(&text[raw_start..i]));
                }
                frags.push(Frag::Tok(id));
                i += len;
                raw_start = i;
                continue;
            }
        }
        i += 1;
    }
    if raw_start < text.len() {
        frags.push(Frag::Raw(&text[raw_start..]));
    }
    frags
}

/// The flavored encode entry point (called from Tokenizer::encode).
pub fn encode_flavored(
    t: &Tokenizer,
    text: &str,
    is_start: bool,
    add_special_tokens: bool,
) -> Result<Vec<i32>, String> {
    let mut out: Vec<i32> = Vec::new();
    if is_start && t.add_bos && t.bos_id >= 0 {
        out.push(t.bos_id);
    }
    let bytes = text.as_bytes();
    let mut is_prev_special = true; // SPM: prefix with space if first fragment is raw
    for frag in split_fragments(t, bytes, add_special_tokens) {
        match &frag {
            Frag::Tok(id) => {
                out.push(*id);
                is_prev_special = true;
            }
            Frag::Raw(raw) => match t.flavor {
                Flavor::Bpe => unreachable!("BPE handled in lib.rs"),
                Flavor::Spm => {
                    let mut text2: Vec<u8> = Vec::with_capacity(raw.len() + 3);
                    if t.add_space_prefix && is_prev_special {
                        text2.push(b' ');
                    }
                    text2.extend_from_slice(raw);
                    escape_whitespace(&mut text2);
                    spm_tokenize(t, &text2, &mut out)?;
                }
                Flavor::Wpm => wpm_tokenize(t, raw, &mut out),
                Flavor::Ugm => ugm_tokenize(t, raw, &mut out),
                Flavor::Rwkv => rwkv_tokenize(t, raw, &mut out),
            },
        }
        if let Frag::Raw(_) = frag {
            is_prev_special = false;
        }
    }
    Ok(out)
}

fn escape_whitespace(text: &mut Vec<u8>) {
    let mut out = Vec::with_capacity(text.len());
    for &b in text.iter() {
        if b == b' ' {
            out.extend_from_slice(ESCAPED_SPACE);
        } else {
            out.push(b);
        }
    }
    *text = out;
}

/// U+2581 ("▁") -> ' ' — used by decode() for metaspace flavors.
pub fn metaspace_to_space(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i..].starts_with(ESCAPED_SPACE) {
            out.push(b' ');
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

/// Decode one piece for metaspace flavors: BYTE-type "<0xXX>" pieces become
/// the raw byte (llama.cpp token_to_piece), "▁" becomes ' '.
pub fn decode_piece(t: &Tokenizer, token: i32) -> Vec<u8> {
    let piece = &t.vocab[token as usize];
    if t.token_types.get(token as usize) == Some(&TT_BYTE)
        && piece.len() == 6
        && piece.starts_with(b"<0x")
        && piece.ends_with(b">")
    {
        if let Ok(v) = u8::from_str_radix(std::str::from_utf8(&piece[3..5]).unwrap_or("zz"), 16) {
            return vec![v];
        }
    }
    metaspace_to_space(piece)
}

fn utf8_len(b0: u8) -> usize {
    if b0 < 0x80 {
        1
    } else if b0 < 0xC0 {
        1 // invalid continuation: consume 1 byte (llama.cpp unicode_len_utf8)
    } else if b0 < 0xE0 {
        2
    } else if b0 < 0xF0 {
        3
    } else {
        4
    }
}

// ---------------------------------------------------------------------------
// SPM (llm_tokenizer_spm_session)
// ---------------------------------------------------------------------------

struct Sym {
    start: usize,
    len: usize,
    prev: i32,
    next: i32,
}

/// Heap order: highest score pops first; ties -> smallest left index first
/// (C++ comparator: lower priority if score lower, or equal score and left larger).
struct Bigram {
    left: i32,
    right: i32,
    score: f32,
    size: usize,
}
impl PartialEq for Bigram {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score && self.left == o.left
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Bigram {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&o.score)
            .unwrap()
            .then_with(|| o.left.cmp(&self.left))
    }
}

#[allow(clippy::too_many_arguments)]
fn spm_try_add(
    t: &Tokenizer,
    text: &[u8],
    symbols: &[Sym],
    queue: &mut BinaryHeap<Bigram>,
    rev_merge: &mut HashMap<Vec<u8>, (i32, i32)>,
    left: i32,
    right: i32,
) {
    if left == -1 || right == -1 {
        return;
    }
    let l = &symbols[left as usize];
    let r = &symbols[right as usize];
    let mut text2 = text[l.start..l.start + l.len].to_vec();
    text2.extend_from_slice(&text[r.start..r.start + r.len]);
    if let Some(&id) = t.piece_map.get(&text2) {
        if (id as usize) < t.vocab.len() {
            queue.push(Bigram {
                left,
                right,
                score: t.scores[id as usize],
                size: text2.len(),
            });
            rev_merge.insert(text2, (left, right));
        }
    }
}

fn spm_resegment(
    t: &Tokenizer,
    text: &[u8],
    symbols: &[Sym],
    rev_merge: &HashMap<Vec<u8>, (i32, i32)>,
    idx: i32,
    out: &mut Vec<i32>,
) -> Result<(), String> {
    let s = &symbols[idx as usize];
    let piece = &text[s.start..s.start + s.len];
    if let Some(&id) = t.piece_map.get(piece) {
        out.push(id);
        return Ok(());
    }
    if let Some(&(l, r)) = rev_merge.get(piece) {
        spm_resegment(t, text, symbols, rev_merge, l, out)?;
        spm_resegment(t, text, symbols, rev_merge, r, out)?;
        return Ok(());
    }
    // byte fallback
    for &b in piece {
        let id = t.byte_tokens[b as usize];
        if id < 0 {
            return Err(format!("spm: no byte-fallback token for byte 0x{b:02x}"));
        }
        out.push(id);
    }
    Ok(())
}

fn spm_tokenize(t: &Tokenizer, text: &[u8], out: &mut Vec<i32>) -> Result<(), String> {
    // split into utf8 symbols
    let mut symbols: Vec<Sym> = Vec::new();
    let mut offs = 0usize;
    let mut index = 0i32;
    while offs < text.len() {
        let l = utf8_len(text[offs]).min(text.len() - offs);
        symbols.push(Sym {
            start: offs,
            len: l,
            prev: index - 1,
            next: if offs + l == text.len() { -1 } else { index + 1 },
        });
        offs += l;
        index += 1;
    }

    let mut rev_merge: HashMap<Vec<u8>, (i32, i32)> = HashMap::new();
    let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();

    for i in 1..symbols.len() as i32 {
        spm_try_add(t, text, &symbols, &mut queue, &mut rev_merge, i - 1, i);
    }

    while let Some(bigram) = queue.pop() {
        let li = bigram.left as usize;
        let ri = bigram.right as usize;
        let (ls_len, rs_len, rs_next) = (symbols[li].len, symbols[ri].len, symbols[ri].next);
        if ls_len == 0 || rs_len == 0 || ls_len + rs_len != bigram.size {
            continue;
        }
        // merge the right symbol into the left one
        symbols[li].len += symbols[ri].len;
        symbols[ri].len = 0;
        symbols[li].next = rs_next;
        if rs_next >= 0 {
            symbols[rs_next as usize].prev = bigram.left;
        }
        let left_prev = symbols[li].prev;
        let left_next = symbols[li].next;
        spm_try_add(t, text, &symbols, &mut queue, &mut rev_merge, left_prev, bigram.left);
        spm_try_add(t, text, &symbols, &mut queue, &mut rev_merge, bigram.left, left_next);
    }

    let mut i = 0i32;
    while i != -1 {
        spm_resegment(t, text, &symbols, &rev_merge, i, out)?;
        i = symbols[i as usize].next;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// WPM (llm_tokenizer_wpm_session) — minimal unicode, documented divergence
// ---------------------------------------------------------------------------

fn wpm_is_whitespace(c: char) -> bool {
    matches!(
        c,
        ' ' | '\t'
            | '\n'
            | '\r'
            | '\u{0B}'
            | '\u{0C}'
            | '\u{85}'
            | '\u{A0}'
            | '\u{1680}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
    ) || ('\u{2000}'..='\u{200A}').contains(&c)
}

fn wpm_is_punctuation(c: char) -> bool {
    if c.is_ascii_punctuation() {
        return true;
    }
    let u = c as u32;
    if (0x2000..=0x206F).contains(&u) {
        // exclude the format/separator chars in this block (ZWSP..ZWJ, LS/PS,
        // LRE..PDF, WJ..NOMS) — emoji clusters must stay inside one word
        return !matches!(u, 0x200B..=0x200F | 0x2028..=0x202E | 0x2060..=0x2064);
    }
    matches!(u, 0x2E00..=0x2E7F | 0x3000..=0x303F | 0xFF00..=0xFFEF)
}

fn is_chinese_char(c: char) -> bool {
    let cpt = c as u32;
    (0x04E00..=0x09FFF).contains(&cpt)
        || (0x03400..=0x04DBF).contains(&cpt)
        || (0x20000..=0x2A6DF).contains(&cpt)
        || (0x2A700..=0x2B73F).contains(&cpt)
        || (0x2B740..=0x2B81F).contains(&cpt)
        || (0x2B920..=0x2CEAF).contains(&cpt)
        || (0x0F900..=0x0FAFF).contains(&cpt)
        || (0x2F800..=0x2FA1F).contains(&cpt)
}

fn wpm_preprocess(t: &Tokenizer, text: &[u8]) -> Vec<Vec<u8>> {
    let s = String::from_utf8_lossy(text);
    let mut words: Vec<Vec<u8>> = vec![Vec::new()];
    for c in s.nfd() {
        if wpm_is_whitespace(c) {
            if !words.last().unwrap().is_empty() {
                words.push(Vec::new());
            }
            continue;
        }
        if c == '\u{0}' || c == '\u{FFFD}' || c.is_control() {
            continue;
        }
        // strip_accents follows lowercase (llama.cpp default): NFD decomposes
        // precomposed accents, then the combining marks are dropped
        if t.wpm_lowercase && (0x0300..=0x036F).contains(&(c as u32)) {
            continue;
        }
        let mapped = if t.wpm_lowercase {
            c.to_lowercase().collect::<String>()
        } else {
            c.to_string()
        };
        if wpm_is_punctuation(c) || is_chinese_char(c) {
            if !words.last().unwrap().is_empty() {
                words.push(Vec::new());
            }
            *words.last_mut().unwrap() = mapped.into_bytes();
            words.push(Vec::new());
        } else {
            words.last_mut().unwrap().extend_from_slice(mapped.as_bytes());
        }
    }
    if words.last().unwrap().is_empty() {
        words.pop();
    }
    words
}

fn wpm_tokenize(t: &Tokenizer, text: &[u8], out: &mut Vec<i32>) {
    for word in wpm_preprocess(t, text) {
        if word.is_empty() {
            continue;
        }
        let mut word1 = ESCAPED_SPACE.to_vec();
        word1.extend_from_slice(&word);
        let n = word1.len();
        let current_tokens = out.len();
        let mut i = 0usize;
        let mut all_matched = true;
        while i < n {
            let mut matched = false;
            let max_j = n.min(i + t.max_token_length as usize + 1);
            let mut j = max_j;
            while j > i {
                if let Some(&id) = t.piece_map.get(&word1[i..j]) {
                    out.push(id);
                    i = j;
                    matched = true;
                    break;
                }
                j -= 1;
            }
            if !matched {
                all_matched = false;
                break;
            }
        }
        if !all_matched {
            out.truncate(current_tokens);
        }
        if out.len() == current_tokens {
            out.push(t.unk_id);
        }
    }
}

// ---------------------------------------------------------------------------
// UGM (llm_tokenizer_ugm_session) — Viterbi + XCDA charsmap
// ---------------------------------------------------------------------------

fn parse_charsmap(blob: &[u8]) -> Result<(Vec<u32>, Vec<u8>), String> {
    if blob.len() < 4 {
        return Ok((Vec::new(), Vec::new()));
    }
    let xcda_blob_size = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
    if 4 + xcda_blob_size > blob.len() || xcda_blob_size % 4 != 0 {
        return Err("charsmap: truncated xcda blob".into());
    }
    let count = xcda_blob_size / 4;
    let mut xcda = Vec::with_capacity(count);
    for k in 0..count {
        xcda.push(u32::from_le_bytes(
            blob[4 + k * 4..4 + k * 4 + 4].try_into().unwrap(),
        ));
    }
    Ok((xcda, blob[4 + xcda_blob_size..].to_vec()))
}

fn xcda_get_base(arr: &[u32], i: usize) -> u32 {
    let n = arr[i];
    (n >> 10) << ((n & (1 << 9)) >> 6)
}
fn xcda_get_lcheck(arr: &[u32], i: usize) -> u32 {
    arr[i] & ((1 << 31) | 0xff)
}
fn xcda_get_leaf(arr: &[u32], i: usize) -> bool {
    (arr[i] >> 8) & 1 == 1
}
fn xcda_get_value(arr: &[u32], i: usize) -> u32 {
    arr[i] & ((1 << 31) - 1)
}

/// normalization step: (replacement bytes, consumed input bytes)
fn ugm_normalize_prefix(t: &Tokenizer, input: &[u8], off: usize) -> (Vec<u8>, usize) {
    if off == input.len() {
        return (Vec::new(), 0);
    }
    // user-defined token longest prefix
    let mut best_ud: Option<&Vec<u8>> = None;
    for piece in &t.ugm_user_defined {
        if input[off..].starts_with(piece)
            && best_ud.map(|b| piece.len() > b.len()).unwrap_or(true)
        {
            best_ud = Some(piece);
        }
    }
    if let Some(p) = best_ud {
        return (p.clone(), p.len());
    }

    let mut longest_prefix_len = 0usize;
    let mut longest_prefix_off = 0usize;
    if !t.ugm_xcda.is_empty() {
        let arr = &t.ugm_xcda;
        let mut node_index = xcda_get_base(arr, 0) as usize;
        for prefix_offset in off..input.len() {
            let c = input[prefix_offset];
            if c == 0 {
                break;
            }
            node_index ^= c as usize;
            if node_index >= arr.len() || xcda_get_lcheck(arr, node_index) != c as u32 {
                break;
            }
            let is_leaf = xcda_get_leaf(arr, node_index);
            node_index ^= xcda_get_base(arr, node_index) as usize;
            if is_leaf {
                longest_prefix_len = prefix_offset - off + 1;
                longest_prefix_off = xcda_get_value(arr, node_index) as usize;
            }
        }
    }
    if longest_prefix_len > 0 {
        if longest_prefix_off >= t.ugm_replacements.len() {
            return (Vec::new(), longest_prefix_len); // defensive: skip bad entry
        }
        let pool = &t.ugm_replacements[longest_prefix_off..];
        let repl_len = pool.iter().position(|&b| b == 0).unwrap_or(pool.len());
        return (pool[..repl_len].to_vec(), longest_prefix_len);
    }
    // valid UTF-8 sequence? consume it, else U+FFFD on 1 byte
    let l = utf8_len(input[off]);
    if off + l <= input.len() && std::str::from_utf8(&input[off..off + l]).is_ok() {
        (input[off..off + l].to_vec(), l)
    } else {
        (b"\xEF\xBF\xBD".to_vec(), 1)
    }
}

fn ugm_normalize(t: &Tokenizer, input: &[u8]) -> Vec<u8> {
    let space: &[u8] = if t.escape_ws { ESCAPED_SPACE } else { b" " };
    let shall_prepend = t.add_space_prefix; // whitespace-as-suffix not supported (default false)
    let shall_merge = t.remove_extra_ws;
    let mut normalized = Vec::with_capacity(input.len() * 3);
    let mut is_space_prepended = false;
    let mut processing_non_ws = false;
    let mut off = 0usize;
    while off < input.len() {
        let (repl, consumed) = ugm_normalize_prefix(t, input, off);
        if consumed == 0 {
            break;
        }
        for &chunk in &repl {
            if chunk != b' ' {
                if !processing_non_ws {
                    processing_non_ws = true;
                    if (shall_prepend && !is_space_prepended) || shall_merge {
                        normalized.extend_from_slice(space);
                        is_space_prepended = true;
                    }
                }
                normalized.push(chunk);
            } else {
                if processing_non_ws {
                    processing_non_ws = false;
                }
                if !shall_merge {
                    normalized.extend_from_slice(space);
                }
            }
        }
        off += consumed;
    }
    normalized
}

#[derive(Clone, Copy)]
struct BestTok {
    token_id: i32,
    input_offset: usize,
    score_sum: f64,
}

fn ugm_tokenize(t: &Tokenizer, text: &[u8], out: &mut Vec<i32>) {
    let normalized = ugm_normalize(t, text);
    let input_len = normalized.len();
    if input_len == 0 {
        return;
    }
    let unknown_score = t.min_score as f64 - 10.0;
    let mut results: Vec<BestTok> = vec![
        BestTok {
            token_id: t.unk_id,
            input_offset: 0,
            score_sum: -f64::MAX,
        };
        input_len + 1
    ];
    results[0].score_sum = 0.0;

    let mut input_offset = 0usize;
    while input_offset < input_len {
        let n_cpt = utf8_len(normalized[input_offset]).min(input_len - input_offset);
        let current_best_score = results[input_offset].score_sum;
        let mut single_codepoint_token_found = false;

        // longest-prefix probing over the piece map (equivalent to the trie
        // walk: a substring absent from the map = a trie dead-end)
        let max_j = input_len.min(input_offset + t.max_token_length as usize);
        let mut j = input_offset + 1;
        while j <= max_j {
            if let Some(&id) = t.piece_map.get(&normalized[input_offset..j]) {
                if j - input_offset == n_cpt {
                    single_codepoint_token_found = true;
                }
                let is_ud = t.token_types.get(id as usize) == Some(&TT_USER_DEFINED);
                let token_score = if is_ud {
                    0.0
                } else {
                    t.scores[id as usize] as f64
                };
                let challenger = current_best_score + token_score;
                let champ = &mut results[j];
                if challenger > champ.score_sum {
                    *champ = BestTok {
                        token_id: id,
                        input_offset,
                        score_sum: challenger,
                    };
                }
            }
            j += 1;
        }

        if !single_codepoint_token_found {
            let challenger = current_best_score + unknown_score;
            let champ = &mut results[input_offset + n_cpt];
            if challenger > champ.score_sum {
                *champ = BestTok {
                    token_id: t.unk_id,
                    input_offset,
                    score_sum: challenger,
                };
            }
        }
        input_offset += n_cpt;
    }

    // backtrack from the end, merging consecutive unknowns; then reverse
    let base = out.len();
    let mut is_prev_unknown = false;
    let mut pos = input_len;
    loop {
        let r = &results[pos];
        let is_unknown = r.token_id == t.unk_id;
        if !(is_prev_unknown && is_unknown) {
            out.push(r.token_id);
        }
        if r.input_offset == 0 {
            break;
        }
        is_prev_unknown = is_unknown;
        pos = r.input_offset;
    }
    out[base..].reverse();
}

// ---------------------------------------------------------------------------
// RWKV (llm_tokenizer_rwkv_session)
// ---------------------------------------------------------------------------

pub fn unescape_rwkv(escaped: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(escaped.len());
    let mut escaping = false;
    let mut hex_remaining = 0u8;
    let mut hex_acc = 0u8;
    for &c in escaped {
        if hex_remaining != 0 {
            let v = if c >= b'a' { c - b'a' + 10 } else { c - b'0' };
            hex_acc = (hex_acc << 4).wrapping_add(v);
            hex_remaining -= 1;
            if hex_remaining == 0 {
                out.push(hex_acc);
                hex_acc = 0;
            }
            continue;
        }
        if escaping {
            match c {
                b't' => out.push(0x09),
                b'n' => out.push(0x0A),
                b'r' => out.push(0x0D),
                b'x' => hex_remaining = 2,
                other => out.push(other),
            }
            escaping = false;
            continue;
        }
        if c == b'\\' {
            escaping = true;
        } else {
            out.push(c);
        }
    }
    out
}

fn rwkv_tokenize(t: &Tokenizer, text: &[u8], out: &mut Vec<i32>) {
    let mut position = 0usize;
    while position < text.len() {
        // greedy longest piece match (the C++ trie walk; ours cannot loop on
        // value-less prefix chains — a sane equivalent)
        let max_j = text.len().min(position + t.max_token_length as usize);
        let mut matched: Option<(i32, usize)> = None;
        let mut j = max_j;
        while j > position {
            if let Some(&id) = t.piece_map.get(&text[position..j]) {
                matched = Some((id, j));
                break;
            }
            j -= 1;
        }
        match matched {
            Some((id, j)) => {
                out.push(id);
                position = j;
            }
            None => {
                out.push(t.unk_id);
                position += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// tests (synthetic vocabs; fixture e2e lives in dllama-gguf's tests)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn spm_tokenizer() -> Tokenizer {
        let vocab_raw: Vec<Vec<u8>> = vec![
            b"<unk>".to_vec(),
            b"<s>".to_vec(),
            b"</s>".to_vec(),
            "▁".as_bytes().to_vec(),
            "▁a".as_bytes().to_vec(),
            "▁b".as_bytes().to_vec(),
            "▁ab".as_bytes().to_vec(),
            b"c".to_vec(),
            "▁abc".as_bytes().to_vec(),
            b"<0x41>".to_vec(),
        ];
        let types = vec![TT_UNKNOWN, TT_CONTROL, TT_CONTROL, TT_NORMAL, TT_NORMAL, TT_NORMAL, TT_NORMAL, TT_NORMAL, TT_NORMAL, TT_BYTE];
        let scores = vec![0.0; 10];
        build_gguf_tokenizer(GgufTokenizerInput {
            model: "llama".into(),
            unk_id: -1,
            vocab_raw,
            scores,
            types,
            bos_id: 1,
            eos_ids: vec![2],
            add_bos: false,
            chat_template: None,
            charsmap: Vec::new(),
            metaspace_add_space_prefix: true,
            metaspace_escape: true,
            remove_extra_ws: true,
            wpm_lowercase: false,
        })
        .unwrap()
    }

    #[test]
    fn spm_merges_and_byte_fallback() {
        let t = spm_tokenizer();
        // " ab" -> escaped "▁ab" -> bigram merge finds token 6
        let toks = t.encode("ab", false, false).unwrap();
        assert_eq!(toks, vec![6], "expected merged ▁ab");
        // byte fallback for unmatched ASCII 'A' (no piece, no bigram).
        // add_space_prefix -> the raw fragment gains a leading space, which
        // escapes to "▁" — a real piece — so output is ▁ + <0x41> (llama.cpp
        // behaves identically).
        let toks = t.encode("A", false, false).unwrap();
        assert_eq!(toks, vec![3, 9], "expected ▁ + <0x41> byte token");
        // special-token splitting: control token matched inline
        let toks = t.encode("ab</s>ab", false, true).unwrap();
        assert_eq!(toks, vec![6, 2, 6]);
    }

    #[test]
    fn spm_decode_converts_metaspace() {
        let t = spm_tokenizer();
        let mut st = crate::DecodeState::default();
        let s = t.decode(&mut st, 6).unwrap();
        assert_eq!(s, " ab", "▁ab must decode to ' ab'");
    }

    fn wpm_tokenizer() -> Tokenizer {
        let vocab_raw: Vec<Vec<u8>> = vec![
            b"[UNK]".to_vec(),
            "▁hello".as_bytes().to_vec(),
            "▁wor".as_bytes().to_vec(),
            b"ld".to_vec(),
            "▁!".as_bytes().to_vec(),
        ];
        let types = vec![TT_UNKNOWN, TT_NORMAL, TT_NORMAL, TT_NORMAL, TT_NORMAL];
        build_gguf_tokenizer(GgufTokenizerInput {
            model: "bert".into(),
            unk_id: -1,
            vocab_raw,
            scores: vec![0.0; 5],
            types,
            bos_id: -1,
            eos_ids: vec![],
            add_bos: false,
            chat_template: None,
            charsmap: Vec::new(),
            metaspace_add_space_prefix: false,
            metaspace_escape: false,
            remove_extra_ws: false,
            wpm_lowercase: true,
        })
        .unwrap()
    }

    #[test]
    fn wpm_wordpiece_and_unk() {
        let t = wpm_tokenizer();
        let toks = t.encode("hello world!", false, false).unwrap();
        // words: "hello", "world", "!" -> ▁hello | (▁wor,##ld) | ▁!
        assert_eq!(toks, vec![1, 2, 3, 4]);
        // an unmatchable word -> [UNK]
        let toks = t.encode("zz", false, false).unwrap();
        assert_eq!(toks, vec![0]);
    }

    fn ugm_tokenizer() -> Tokenizer {
        let vocab_raw: Vec<Vec<u8>> = vec![
            b"<unk>".to_vec(),
            "▁".as_bytes().to_vec(),
            "ab".as_bytes().to_vec(),
            "c".as_bytes().to_vec(),
            "▁ab".as_bytes().to_vec(),
            "▁c".as_bytes().to_vec(),
        ];
        let types = vec![TT_UNKNOWN, TT_NORMAL, TT_NORMAL, TT_NORMAL, TT_NORMAL, TT_NORMAL];
        // scores: ▁=0, ab=-1, c=-2, ▁ab=-0.5, ▁c=-1.0
        let scores = vec![0.0, 0.0, -1.0, -2.0, -0.5, -1.0];
        build_gguf_tokenizer(GgufTokenizerInput {
            model: "t5".into(),
            unk_id: -1,
            vocab_raw,
            scores,
            types,
            bos_id: -1,
            eos_ids: vec![],
            add_bos: false,
            chat_template: None,
            charsmap: Vec::new(),
            metaspace_add_space_prefix: true,
            metaspace_escape: true,
            remove_extra_ws: false,
            wpm_lowercase: false,
        })
        .unwrap()
    }

    #[test]
    fn ugm_viterbi_picks_best_path() {
        let t = ugm_tokenizer();
        // "ab c" -> prepend space + escape -> "▁ab▁c"
        // best path: ▁ab (-0.5) + ▁c (-1.0) = -1.5
        //   vs ▁ab + ▁ (0) + c (-2.0) = -2.5 -> picks ▁ab, ▁c
        let toks = t.encode("ab c", false, false).unwrap();
        assert_eq!(toks, vec![4, 5], "Viterbi must pick the better path");
    }

    fn rwkv_tokenizer() -> Tokenizer {
        // escaped pieces: "▁" = \\xe2\\x96\\x81, "hello" plain, "a b" = a\\x20b
        let vocab_raw: Vec<Vec<u8>> = vec![
            b"<unk>".to_vec(),
            b"\\xe2\\x96\\x81".to_vec(),
            b"hello".to_vec(),
            b"a\\x20b".to_vec(),
        ];
        let types = vec![TT_UNKNOWN, TT_NORMAL, TT_NORMAL, TT_NORMAL];
        build_gguf_tokenizer(GgufTokenizerInput {
            model: "rwkv".into(),
            unk_id: -1,
            vocab_raw,
            scores: vec![0.0; 4],
            types,
            bos_id: -1,
            eos_ids: vec![],
            add_bos: false,
            chat_template: None,
            charsmap: Vec::new(),
            metaspace_add_space_prefix: false,
            metaspace_escape: false,
            remove_extra_ws: false,
            wpm_lowercase: false,
        })
        .unwrap()
    }

    #[test]
    fn rwkv_unescape_and_match() {
        let t = rwkv_tokenizer();
        // vocab[3] unescapes to "a b" — must match the raw text
        let toks = t.encode("a b", false, false).unwrap();
        assert_eq!(toks, vec![3]);
        // unmatched -> unk, advances 1 byte
        let toks = t.encode("zz", false, false).unwrap();
        assert_eq!(toks, vec![0, 0]);
    }
}
