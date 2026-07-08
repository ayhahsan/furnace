// Token sampling: repeat penalty -> temperature -> top-k -> top-p ->
// renormalize -> draw. The order is fixed; changing it changes the
// distribution being sampled.

/// Minimal PCG32 (O'Neill). Seed fully determines the output stream.
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64) -> Pcg32 {
        let mut rng = Pcg32 { state: 0, inc: (0xda3e_39cb_94b9_5bdb << 1) | 1 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1) with 24 bits of precision (f32 mantissa).
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// llama.cpp convention: positive logits are divided by the penalty,
/// negative ones multiplied -- both directions mean "less likely".
/// Applied once per unique recent token.
pub(crate) fn apply_repeat_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut seen: Vec<u32> = Vec::with_capacity(recent.len());
    for &t in recent {
        if !seen.contains(&t) {
            seen.push(t);
            let l = &mut logits[t as usize];
            if *l > 0.0 {
                *l /= penalty;
            } else {
                *l *= penalty;
            }
        }
    }
}

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,       // 0 disables the cut
    pub top_p: f32,         // 1.0 disables the cut
    pub repeat_penalty: f32, // 1.0 disables the penalty
    pub repeat_window: usize,
    rng: Pcg32,
    draws: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, repeat_penalty: f32, seed: u64) -> Sampler {
        Sampler {
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            repeat_window: 64,
            rng: Pcg32::new(seed),
            draws: 0,
        }
    }

    /// RNG draws taken so far. Greedy (temperature 0 or top_k 1) must never
    /// advance this.
    pub fn draws(&self) -> u64 {
        self.draws
    }

    pub fn sample(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        let _t = crate::perf::time(&crate::perf::SAMPLE);
        let mut lg = logits.to_vec();
        apply_repeat_penalty(&mut lg, recent, self.repeat_penalty);

        // pure argmax short-circuit: no RNG state is touched
        if self.temperature == 0.0 || self.top_k == 1 {
            return crate::generate::argmax(&lg);
        }

        let candidates = self.filtered(&lg);
        let u = self.rng.next_f32();
        self.draws += 1;

        let mut cum = 0.0;
        for &(id, p) in &candidates {
            cum += p;
            if u < cum {
                return id;
            }
        }
        candidates.last().expect("empty candidate set").0
    }

    /// Temperature + top-k + softmax + top-p + renormalize, on logits that
    /// already carry the repeat penalty. Returns (id, prob) sorted by
    /// descending probability, summing to 1.
    pub(crate) fn filtered(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        let mut cand: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as u32, l / self.temperature))
            .collect();

        if self.top_k > 0 && self.top_k < cand.len() {
            cand.select_nth_unstable_by(self.top_k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
            cand.truncate(self.top_k);
        }
        cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // softmax over the surviving candidates only, max-subtracted
        let max = cand[0].1;
        let mut sum = 0.0;
        for c in &mut cand {
            c.1 = (c.1 - max).exp();
            sum += c.1;
        }
        for c in &mut cand {
            c.1 /= sum;
        }

        if self.top_p < 1.0 {
            let mut cum = 0.0;
            let mut keep = cand.len();
            for (i, c) in cand.iter().enumerate() {
                cum += c.1;
                if cum >= self.top_p {
                    keep = i + 1; // the crossing token is included
                    break;
                }
            }
            cand.truncate(keep);
            let s: f32 = cand.iter().map(|c| c.1).sum();
            for c in &mut cand {
                c.1 /= s;
            }
        }
        cand
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(temperature: f32, top_k: usize, top_p: f32) -> Sampler {
        Sampler::new(temperature, top_k, top_p, 1.0, 7)
    }

    fn assert_close(got: f32, want: f32) {
        assert!((got - want).abs() < 1e-5, "got {}, want {}", got, want);
    }

    #[test]
    fn temperature_reshapes_the_distribution() {
        // logits [0, ln 3]: at T=1 probs are [0.25, 0.75].
        // At T=0.5 gaps double, ratios square: [1:9] -> [0.1, 0.9].
        let logits = vec![0.0, 3.0f32.ln()];
        let c = sampler(1.0, 0, 1.0).filtered(&logits);
        assert_eq!(c[0].0, 1);
        assert_close(c[0].1, 0.75);
        assert_close(c[1].1, 0.25);

        let c = sampler(0.5, 0, 1.0).filtered(&logits);
        assert_close(c[0].1, 0.9);
        assert_close(c[1].1, 0.1);
    }

    #[test]
    fn top_k_keeps_exactly_k() {
        let logits = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        let c = sampler(1.0, 2, 1.0).filtered(&logits);
        let ids: Vec<u32> = c.iter().map(|c| c.0).collect();
        assert_eq!(ids, vec![0, 2]); // the two largest, sorted desc
        assert_close(c.iter().map(|c| c.1).sum::<f32>(), 1.0);
    }

    #[test]
    fn top_p_cuts_at_the_crossing_token() {
        // probs 0.5 / 0.3 / 0.15 / 0.05; p=0.8 -> cumulative hits 0.8 at the
        // second token, so exactly two survive, renormalized to 0.625/0.375
        let logits: Vec<f32> = [0.5f32, 0.3, 0.15, 0.05].iter().map(|p| p.ln()).collect();
        let c = sampler(1.0, 0, 0.8).filtered(&logits);
        assert_eq!(c.len(), 2);
        assert_eq!((c[0].0, c[1].0), (0, 1));
        assert_close(c[0].1, 0.625);
        assert_close(c[1].1, 0.375);
    }

    #[test]
    fn repeat_penalty_follows_llama_cpp_sign_convention() {
        let mut logits = vec![2.0, -2.0, 1.0];
        // ids 0 and 1 are recent (id 0 twice: penalized once, not twice)
        apply_repeat_penalty(&mut logits, &[0, 1, 0], 2.0);
        assert_eq!(logits[0], 1.0); // positive: divided
        assert_eq!(logits[1], -4.0); // negative: multiplied
        assert_eq!(logits[2], 1.0); // not recent: untouched
    }

    #[test]
    fn degenerate_filters_disable_cleanly() {
        // top_k=0 and top_p=1.0: full vocab survives, probs sum to 1
        let logits = vec![1.0, 0.5, 0.0, -0.5];
        let c = sampler(1.0, 0, 1.0).filtered(&logits);
        assert_eq!(c.len(), 4);
        assert_close(c.iter().map(|c| c.1).sum::<f32>(), 1.0);
    }

    #[test]
    fn temperature_zero_is_argmax_with_no_rng_draw() {
        let logits = vec![0.1, 3.0, 0.2];
        let mut s = sampler(0.0, 40, 0.9);
        assert_eq!(s.sample(&logits, &[]), 1);
        assert_eq!(s.draws(), 0);
        // top_k == 1 short-circuits too
        let mut s = sampler(0.7, 1, 0.9);
        assert_eq!(s.sample(&logits, &[]), 1);
        assert_eq!(s.draws(), 0);
        // a real sampling call does draw
        let mut s = sampler(0.7, 40, 0.9);
        s.sample(&logits, &[]);
        assert_eq!(s.draws(), 1);
    }

    #[test]
    fn same_seed_same_stream_different_seed_different_stream() {
        // near-uniform logits so different streams almost surely diverge
        let logits: Vec<f32> = (0..100).map(|i| (i % 7) as f32 * 0.01).collect();
        let run = |seed: u64| -> Vec<u32> {
            let mut s = Sampler::new(1.0, 0, 1.0, 1.0, seed);
            (0..20).map(|_| s.sample(&logits, &[])).collect()
        };
        assert_eq!(run(42), run(42));
        assert_ne!(run(42), run(43));
    }
}
