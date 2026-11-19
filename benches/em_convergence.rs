use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use epistat_rng::Stream;
use latent_class::{EmConfig, IndicatorMatrix, LcaModel};

fn generate_synthetic_data(n: usize, j: usize, k_true: usize, seed: u64) -> IndicatorMatrix {
    let mut rng = Stream::seeded(seed);
    let mut cols = Vec::with_capacity(j);

    // Create J indicator columns
    for jj in 0..j {
        let mut col = Vec::with_capacity(n);
        // Randomly assign class-specific probabilities for this indicator
        let class_probs: Vec<f64> = (0..k_true).map(|_| 0.1 + 0.8 * rng.uniform()).collect();

        for i in 0..n {
            let class = i % k_true;
            let p = class_probs[class];
            col.push(Some(u8::from(rng.uniform() < p)));
        }
        cols.push((format!("v{jj}"), col));
    }

    IndicatorMatrix::from_columns(cols).unwrap()
}

fn bench_em_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("EM Convergence");

    // N = {1_000, 10_000}, K = {3, 6}, J = 20
    for &n in &[1_000, 10_000] {
        for &k in &[3, 6] {
            let j = 20;
            let data = generate_synthetic_data(n, j, k, 42);
            let model = LcaModel::new(k).unwrap().with_n_restarts(1); // Benchmarking single restart

            group.bench_with_input(
                BenchmarkId::new("Single Restart", format!("N{n}K{k}J{j}")),
                &data,
                |b, d| {
                    b.iter(|| model.fit(d).unwrap());
                },
            );
        }
    }

    // Parallel speedup check at N=10_000, K=6, J=20
    let n = 10_000;
    let k = 6;
    let j = 20;
    let data = generate_synthetic_data(n, j, k, 42);

    let sequential_config = EmConfig {
        n_restarts: 8,
        ..Default::default()
    };
    let model = LcaModel::new(k).unwrap().with_config(sequential_config);

    group.bench_function(format!("Parallel Restarts (8) N{n}K{k}"), |b| {
        b.iter(|| model.fit(&data).unwrap());
    });

    group.finish();
}

criterion_group!(benches, bench_em_convergence);
criterion_main!(benches);
