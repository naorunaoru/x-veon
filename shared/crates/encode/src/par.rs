//! Per-pixel loops in fixed-size chunks: on a pool of `threads` threads with the `threading`
//! feature (the native build), in order without it (WASM). Each chunk writes only its own
//! outputs from its own inputs, so the bytes never depend on the thread count.
pub const CHUNK: usize = 3 * 64 * 1024;
pub fn fill<T: Send>(out: &mut [T], threads: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    #[cfg(feature = "threading")]
    if threads > 1 {
        use rayon::prelude::*;
        if let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
            pool.install(|| out.par_chunks_mut(CHUNK).enumerate().for_each(|(i, chunk)| f(i * CHUNK, chunk)));
            return;
        }
    }
    let _ = threads;
    for (i, chunk) in out.chunks_mut(CHUNK).enumerate() { f(i * CHUNK, chunk); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn threads_produce_identical_elements() {
        let mut one = vec![0u32; 1_000_003];
        let mut eight = one.clone();
        let f = |start, chunk: &mut [u32]| {
            for (k, v) in chunk.iter_mut().enumerate() { *v = (start + k) as u32 * 3; }
        };
        fill(&mut one, 1, f);
        fill(&mut eight, 8, f);
        assert_eq!(one, eight);
        for (i, v) in one.iter().enumerate() { assert_eq!(*v, i as u32 * 3); }
    }
    #[test]
    fn chunks_start_on_whole_pixels() {
        for threads in [1, 8] {
            let mut out = vec![0u8; 1_000_003];
            fill(&mut out, threads, |start, _| assert_eq!(start % 3, 0));
        }
    }
}
