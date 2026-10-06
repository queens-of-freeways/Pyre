//! Token sampling (G6.2): temperature / top-k / top-p with a deterministic
//! RNG. `temperature <= 0` is greedy and **must be bit-identical with
//! `dllama_exec::argmax`** (first-occurrence tie-break) — that keeps the
//! default CLI and the parity workflows unchanged.
//!
//! Pipeline when sampling: `logits / temperature` → softmax (the bit-exact
//! kernel) → top-k keep → top-p (nucleus) cut → renormalize → cumulative
//! sample. The order matches the C++ sampler (temperature before softmax,
//! top-k before top-p).

/// xorshift64* — small, deterministic, seedable. Not cryptographic; fine for
/// token sampling.
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // avoid the all-zero fixed point
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    pub fn from_time() -> Self {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        Self::new(t.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[derive(Clone)]
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub rng: Rng,
}

impl Sampler {
    /// Greedy (bit-identical with argmax) — the default everywhere.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            rng: Rng::new(0),
        }
    }

    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64, time_seed: bool) -> Self {
        Self {
            temperature: if temperature.is_finite() && temperature > 0.0 {
                temperature
            } else {
                0.0
            },
            top_k,
            top_p: if top_p.is_finite() && top_p > 0.0 && top_p < 1.0 { top_p } else { 1.0 },
            rng: if time_seed { Rng::from_time() } else { Rng::new(seed) },
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Pick the next token id from the logits over the vocab.
    pub fn next(&mut self, logits: &[f32]) -> i32 {
        if self.is_greedy() {
            return dllama_exec::argmax(logits) as i32;
        }
        // temperature + softmax over the full vocab (one row)
        let mut probs: Vec<f32> = logits
            .iter()
            .map(|&l| l / self.temperature)
            .collect();
        let len = probs.len();
        dllama_kernel::cpu::softmax(&mut probs, len);

        // candidate list: index + prob, sorted by prob desc (stable on ties
        // by index so runs are reproducible)
        let mut cand: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        cand.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let mut keep = len;
        if self.top_k > 0 && self.top_k < len {
            keep = self.top_k;
        }
        if self.top_p < 1.0 {
            let mut acc = 0.0f32;
            let mut cut = 0usize;
            for (i, &(_, p)) in cand.iter().enumerate().take(keep) {
                acc += p;
                cut = i + 1;
                if acc >= self.top_p {
                    break;
                }
            }
            keep = cut.max(1);
        }
        let kept = &cand[..keep];
        let sum: f32 = kept.iter().map(|&(_, p)| p).sum();
        if !(sum > 0.0) {
            return kept[0].0 as i32;
        }
        // renormalize + cumulative sample
        let r = self.rng.next_f32() * sum;
        let mut acc = 0.0f32;
        for &(i, p) in kept {
            acc += p;
            if r < acc {
                return i as i32;
            }
        }
        kept[kept.len() - 1].0 as i32
    }
}

/// Parse sampler flags from a CLI arg list.
pub fn sampler_from_args(args: &[String]) -> Sampler {
    let f = crate::g1::flag;
    let temperature: f32 = f(args, "--temperature").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let top_k: usize = f(args, "--top-k").and_then(|v| v.parse().ok()).unwrap_or(0);
    let top_p: f32 = f(args, "--top-p").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let seed: u64 = f(args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(0);
    if temperature <= 0.0 {
        Sampler::greedy()
    } else {
        Sampler::new(temperature, top_k, top_p, seed, f(args, "--seed").is_none())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..n).map(|i| ((rng.next_u64() % 97) as f32 / 10.0) - (i as f32 % 3.0)).collect()
    }

    #[test]
    fn greedy_matches_argmax() {
        let l = logits(1000, 7);
        let mut s = Sampler::greedy();
        assert_eq!(s.next(&l) as usize, dllama_exec::argmax(&l));
        // explicit temperature 0 or negative must stay greedy
        let mut s2 = Sampler::new(0.0, 40, 0.9, 1, false);
        assert_eq!(s2.next(&l) as usize, dllama_exec::argmax(&l));
    }

    #[test]
    fn top_k_one_is_argmax() {
        let l = logits(500, 3);
        let mut s = Sampler::new(0.7, 1, 1.0, 9, false);
        assert_eq!(s.next(&l) as usize, dllama_exec::argmax(&l));
    }

    #[test]
    fn tiny_top_p_is_argmax() {
        let l = logits(500, 5);
        let mut s = Sampler::new(1.0, 0, 1e-9, 9, false);
        assert_eq!(s.next(&l) as usize, dllama_exec::argmax(&l));
    }

    #[test]
    fn sampled_token_always_in_top_k() {
        let l = logits(2000, 11);
        let mut top: Vec<usize> = (0..2000).collect();
        top.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
        let top: std::collections::HashSet<usize> = top[..10].iter().copied().collect();
        let mut s = Sampler::new(1.5, 10, 1.0, 123, false);
        for _ in 0..200 {
            let t = s.next(&l) as usize;
            assert!(top.contains(&t), "sampled outside top-k: {t}");
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let l = logits(300, 13);
        let mut a = Sampler::new(0.8, 0, 0.95, 77, false);
        let mut b = Sampler::new(0.8, 0, 0.95, 77, false);
        for _ in 0..50 {
            assert_eq!(a.next(&l), b.next(&l));
        }
    }

    #[test]
    fn rng_uniform_in_range() {
        let mut rng = Rng::new(5);
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
