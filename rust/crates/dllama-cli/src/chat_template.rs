//! Chat prompt rendering (G6.2): the GGUF `chat_template` (Jinja) rendered
//! via minijinja, with an arch fallback table, and plain ChatML as the last
//! resort. llama.cpp templates commonly use `strftime_now` /
//! `raise_exception` / `add_generation_prompt` â€” shims and globals are
//! provided so most stock templates render unmodified.

use minijinja::{context, Environment};

pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

// ---------------------------------------------------------------------------
// minja compatibility: llama.cpp templates use Python-style string METHODS
// (`x.startswith(y)`, `x.split(sep).lstrip('\n')`) that minijinja doesn't
// implement. We rewrite method chains into filter chains — `x | split(sep)`
// is semantically identical — and register the method set as filters.
// ---------------------------------------------------------------------------

/// Method names llama.cpp's `minja` supports that we rewrite to filters.
const MINJA_METHODS: &[&str] = &[
    "startswith", "endswith", "split", "lstrip", "rstrip", "strip", "title", "lower", "upper",
];

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Walk backward from `dot_pos` (index of the '.') to find the start of the
/// receiver expression: a chain of identifiers, `.field`, balanced
/// `(...)`/`[...]` groups, and filter `|` separators. Stops at operators,
/// commas, and spaces outside groups.
fn receiver_start(t: &[char], dot_pos: usize) -> usize {
    let mut i = dot_pos;
    while i > 0 {
        let prev = t[i - 1];
        if is_ident_char(prev) {
            i -= 1;
            continue;
        }
        if prev == ')' || prev == ']' {
            let (open, close) = if prev == ')' { ('(', ')') } else { ('[', ']') };
            let mut depth = 0i32;
            i -= 1;
            while i > 0 {
                if t[i - 1] == close {
                    depth += 1;
                } else if t[i - 1] == open {
                    depth -= 1;
                    if depth == 0 {
                        i -= 1;
                        break;
                    }
                }
                i -= 1;
            }
            continue;
        }
        if prev == '|' || prev == '.' {
            i -= 1;
            while i > 0 && t[i - 1] == ' ' {
                i -= 1;
            }
            continue;
        }
        break;
    }
    i
}

/// Rewrite `recv.meth(args)` -> `recv | meth(args)` for the minja method set.
/// Runs to a fixpoint so chains (`a.split(s).lstrip(n)`) fully convert
/// (`a | split(s) | lstrip(n)`). Anything unrewritable is left in place —
/// the render simply fails and the fallback chain takes over.
pub fn normalize_minja_template(src: &str) -> String {
    // one rewrite per pass with a full rebuild: coordinates stay consistent
    // and every pass removes one `.method(` occurrence, so this terminates.
    let mut out = src.to_string();
    'outer: loop {
        let chars: Vec<char> = out.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            if chars[i] == '.' && i + 1 < chars.len() && is_ident_char(chars[i + 1]) {
                let mut j = i + 1;
                while j < chars.len() && is_ident_char(chars[j]) {
                    j += 1;
                }
                let name: String = chars[i + 1..j].iter().collect();
                if MINJA_METHODS.contains(&name.as_str()) && j < chars.len() && chars[j] == '(' {
                    let mut depth = 0i32;
                    let mut k = j;
                    while k < chars.len() {
                        if chars[k] == '(' {
                            depth += 1;
                        } else if chars[k] == ')' {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        k += 1;
                    }
                    if k >= chars.len() {
                        // unbalanced — leave as-is; the render will fail and
                        // the fallback chain takes over
                        i = j;
                        continue;
                    }
                    let start = receiver_start(&chars, i);
                    let mut next = String::with_capacity(out.len());
                    for c in &chars[..start] {
                        next.push(*c);
                    }
                    for c in &chars[start..i] {
                        next.push(*c);
                    }
                    next.push_str(" | ");
                    next.push_str(&name);
                    for c in &chars[j..=k] {
                        next.push(*c);
                    }
                    for c in &chars[k + 1..] {
                        next.push(*c);
                    }
                    out = next;
                    continue 'outer;
                }
            }
            i += 1;
        }
        return out;
    }
}


/// Python str.strip semantics: remove any leading/trailing chars present
/// in the set (not repeated whole-pattern matches).
fn strip_set(s: &str, chars: &str, start: bool, end: bool) -> String {
    let set: std::collections::HashSet<char> = chars.chars().collect();
    let mut out = s;
    if start {
        out = out.trim_start_matches(|c| set.contains(&c));
    }
    if end {
        out = out.trim_end_matches(|c| set.contains(&c));
    }
    out.to_string()
}

fn add_minja_filters(env: &mut Environment<'_>) {
    use minijinja::value::Value;
    // string filters with Python method semantics
    env.add_filter("startswith", |s: String, prefix: String| s.starts_with(&prefix));
    env.add_filter("endswith", |s: String, suffix: String| s.ends_with(&suffix));
    env.add_filter("split", |s: String, sep: Option<String>| match sep {
        Some(sep) => Value::from_iter(s.split(&sep).map(|p| p.to_string())),
        None => Value::from_iter(s.split_whitespace().map(|p| p.to_string())),
    });
    env.add_filter("lstrip", |s: String, chars: Option<String>| match chars {
        Some(c) => strip_set(&s, &c, true, false),
        None => s.trim_start().to_string(),
    });
    env.add_filter("rstrip", |s: String, chars: Option<String>| match chars {
        Some(c) => strip_set(&s, &c, false, true),
        None => s.trim_end().to_string(),
    });
    env.add_filter("strip", |s: String, chars: Option<String>| match chars {
        Some(c) => strip_set(&s, &c, true, true),
        None => s.trim().to_string(),
    });
    env.add_filter("title", |s: String| {
        let mut out = String::with_capacity(s.len());
        let mut cap = true;
        for c in s.chars() {
            if c.is_ascii_alphabetic() && cap {
                out.extend(c.to_uppercase());
                cap = false;
            } else if !c.is_ascii_alphanumeric() {
                out.push(c);
                cap = true;
            } else {
                out.push(c);
            }
        }
        out
    });
    env.add_filter("lower", |s: String| s.to_lowercase());
    env.add_filter("upper", |s: String| s.to_uppercase());
}

/// Self-contained ChatML (Qwen family; also the historical default).
const TMPL_CHATML: &str = concat!(
    "{%- for m in messages %}",
    "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n",
    "{%- endfor %}",
    "<|im_start|>assistant\n"
);

/// Self-contained Llama-3 headers (no bos var â€” the tokenizer adds BOS).
const TMPL_LLAMA3: &str = concat!(
    "{%- for m in messages %}",
    "<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content | trim }}<|eot_id|>",
    "{%- endfor %}",
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
);

/// Mistral [INST]/[/INST] (no system role support in this minimal form).
const TMPL_MISTRAL: &str = concat!(
    "{%- for m in messages %}",
    "{%- if m.role == 'user' %}[INST] {{ m.content }} [/INST]",
    "{%- elif m.role == 'assistant' %}{{ m.content }}</s>",
    "{%- endif %}",
    "{%- endfor %}"
);

fn render_jinja(template: &str, messages: &[ChatMsg]) -> Result<String, String> {
    let template = normalize_minja_template(template);
    let mut env = Environment::new();
    // llama.cpp templates expect the final newline (assistant header ends \n)
    env.set_keep_trailing_newline(true);
    add_minja_filters(&mut env);
    // llama.cpp template shims
    env.add_function("strftime_now", |_fmt: String| -> String { "2026-01-01".into() });
    env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", template.as_str())
        .map_err(|e| format!("template parse: {e}"))?;
    let tmpl = env.get_template("chat").map_err(|e| e.to_string())?;
    let msgs: Vec<_> = messages
        .iter()
        .map(|m| context! { role => m.role, content => m.content })
        .collect();
    let ctx = context! {
        messages => msgs,
        add_generation_prompt => true,
        bos_token => "",
        eos_token => "",
    };
    tmpl.render(ctx).map_err(|e| format!("template render: {e}"))
}

fn chatml_plain(messages: &[ChatMsg]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn fallback_template(arch: dllama_ir::Arch) -> Option<&'static str> {
    match arch {
        dllama_ir::Arch::Qwen3 | dllama_ir::Arch::Qwen3Moe => Some(TMPL_CHATML),
        dllama_ir::Arch::Llama => Some(TMPL_LLAMA3),
    }
}

/// Render the chat prompt. Chain: GGUF `chat_template` â†’ arch table â†’ plain
/// ChatML. Returns the prompt and which source produced it (for a startup
/// log line).
pub fn render_chat_prompt(
    template: Option<&str>,
    arch: dllama_ir::Arch,
    messages: &[ChatMsg],
) -> (String, &'static str) {
    if let Some(t) = template.filter(|t| !t.trim().is_empty()) {
        match render_jinja(t, messages) {
            Ok(s) if !s.is_empty() => return (s, "gguf chat_template"),
            Ok(_) => {}
            Err(e) => eprintln!("[chat-template] GGUF template failed ({e}); using arch fallback"),
        }
    }
    if let Some(t) = fallback_template(arch) {
        if let Ok(s) = render_jinja(t, messages) {
            return (s, "arch fallback template");
        } else {
            eprintln!("[chat-template] arch fallback also failed");
        }
    }
    (chatml_plain(messages), "plain ChatML")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs() -> Vec<ChatMsg> {
        vec![
            ChatMsg { role: "system".into(), content: "You are helpful.".into() },
            ChatMsg { role: "user".into(), content: "hi".into() },
        ]
    }

    #[test]
    fn chatml_fallback_shape() {
        let (s, src) = render_chat_prompt(None, dllama_ir::Arch::Qwen3, &msgs());
        assert_eq!(src, "arch fallback template");
        assert!(s.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama_arch_uses_llama3_headers() {
        let (s, src) = render_chat_prompt(None, dllama_ir::Arch::Llama, &msgs());
        assert_eq!(src, "arch fallback template");
        assert!(s.contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"));
        assert!(s.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn gguf_template_wins_when_valid() {
        let t = "{%- for m in messages %}[{{ m.role }}:{{ m.content }}]{%- endfor %}ASSIST:";
        let (s, src) = render_chat_prompt(Some(t), dllama_ir::Arch::Qwen3, &msgs());
        assert_eq!(src, "gguf chat_template");
        assert_eq!(s, "[system:You are helpful.][user:hi]ASSIST:");
    }

    #[test]
    fn broken_gguf_template_falls_back() {
        let t = "{%- if oops";
        let (s, src) = render_chat_prompt(Some(t), dllama_ir::Arch::Qwen3, &msgs());
        assert_eq!(src, "arch fallback template");
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn minja_method_calls_render() {
        // exact constructs from the Qwen3 official template: .split().lstrip()
        // chains, .strip(), .startswith() on message content
        let t = "{%- for m in messages %}{%- set c = (m.content.split('x')|last).lstrip('n') %}{{ c.strip('n') }}{{ c.startswith('ab') }}{%- endfor %}";
        let (s2, src) = render_chat_prompt(
            Some(t),
            dllama_ir::Arch::Qwen3,
            &[ChatMsg { role: "user".into(), content: "zzxabc".into() }],
        );
        assert_eq!(src, "gguf chat_template");
        assert_eq!(s2, "abcTrue");
    }

    #[test]
    fn normalize_keeps_plain_jinja() {
        // templates without method calls must pass through untouched
        let t = "{{ messages[0].role }}";
        assert_eq!(normalize_minja_template(t), t);
        // chains fully rewrite
        assert_eq!(
            normalize_minja_template("a.split('x').lstrip('n')"),
            "a | split('x') | lstrip('n')"
        );
        // dict- access receivers work
        assert_eq!(
            normalize_minja_template("m['content'].startswith('a')"),
            "m['content'] | startswith('a')"
        );
    }

    #[test]
    fn add_generation_prompt_global_available() {
        // many stock templates gate the assistant header on this global
        let t = "{% if add_generation_prompt %}GO{% endif %}";
        let (s, _) = render_chat_prompt(Some(t), dllama_ir::Arch::Qwen3, &msgs());
        assert_eq!(s, "GO");
    }
}
