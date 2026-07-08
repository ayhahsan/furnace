// Cheap accumulating op timers, enabled by `furnace run --timings`.
// When disabled, each guard costs one relaxed atomic load and no clock read.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn enable() {
    ENABLED.store(true, Relaxed);
}

pub struct Counter {
    name: &'static str,
    nanos: AtomicU64,
    calls: AtomicU64,
}

impl Counter {
    const fn new(name: &'static str) -> Counter {
        Counter { name, nanos: AtomicU64::new(0), calls: AtomicU64::new(0) }
    }
}

pub static MATMUL: Counter = Counter::new("matmul");
pub static RMSNORM: Counter = Counter::new("rmsnorm");
pub static SOFTMAX: Counter = Counter::new("softmax");
pub static ROPE: Counter = Counter::new("rope");
pub static ATTN_DOT: Counter = Counter::new("attn scores");
pub static ATTN_APPLY: Counter = Counter::new("attn apply");
pub static ELEMENTWISE: Counter = Counter::new("add/swiglu");
pub static EMBED: Counter = Counter::new("embed");
pub static SAMPLE: Counter = Counter::new("sample");

static ALL: [&Counter; 9] = [
    &MATMUL, &RMSNORM, &SOFTMAX, &ROPE, &ATTN_DOT, &ATTN_APPLY,
    &ELEMENTWISE, &EMBED, &SAMPLE,
];

/// RAII guard: measures from creation to drop when timing is enabled.
pub struct Guard {
    counter: &'static Counter,
    start: Option<Instant>,
}

pub fn time(counter: &'static Counter) -> Guard {
    let start = ENABLED.load(Relaxed).then(Instant::now);
    Guard { counter, start }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            self.counter.nanos.fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
            self.counter.calls.fetch_add(1, Relaxed);
        }
    }
}

/// Zero all counters (called after prefill so the report covers decode only).
pub fn reset() {
    for c in ALL {
        c.nanos.store(0, Relaxed);
        c.calls.store(0, Relaxed);
    }
}

/// Print the per-op breakdown against a measured wall time.
pub fn report(label: &str, wall_ms: f64) {
    if !ENABLED.load(Relaxed) {
        return;
    }
    eprintln!("{} op breakdown ({:.0} ms wall):", label, wall_ms);
    let mut accounted = 0.0;
    for c in ALL {
        let ms = c.nanos.load(Relaxed) as f64 / 1e6;
        let calls = c.calls.load(Relaxed);
        if calls == 0 {
            continue;
        }
        accounted += ms;
        eprintln!(
            "  {:12} {:9.1} ms  {:5.1}%  {:7} calls",
            c.name,
            ms,
            100.0 * ms / wall_ms,
            calls
        );
    }
    eprintln!(
        "  {:12} {:9.1} ms  {:5.1}%",
        "unaccounted",
        wall_ms - accounted,
        100.0 * (wall_ms - accounted) / wall_ms
    );
}
