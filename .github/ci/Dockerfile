# CI image: every check runs in this Dockerfile under BuildKit (docker/setup-buildx-action). The
# aarch64 stages (arm64, parity-arm64, fuzz on arm64) build natively on ubuntu-24.04-arm.
# The dependencies (rayon) are fetched in a layer before the source is copied, from the manifest
# and lockfile with a stub lib, so a source change does not fetch again.
# Every tool is installed in a stage before `COPY . .`, so a new commit does not install it again.
FROM rust:1.99.1-bookworm AS toolchain
RUN rustup component add rustfmt clippy && rustup target add aarch64-unknown-linux-gnu
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && touch src/lib.rs
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry \
    cargo fetch --locked
RUN rm -r src

FROM toolchain AS src
COPY . .
# The target dir is a cache mount shared by every commit and branch, and cargo decides freshness by
# mtime: a checkout older than artifacts another commit built would reuse them (parity-arm64 once
# ran a stale oracle). Future mtimes make cargo rebuild this crate in every RUN; deps stay cached.
RUN find src examples -name '*.rs' -exec touch -d '2100-01-01' {} + && touch -d '2100-01-01' Cargo.toml

# The checks run in the build, so a red check fails the build; a rerun of the same source is a
# BuildKit cache hit. The registry and target dirs are cache mounts, so they persist across builds.
FROM src AS ci
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    cargo fmt --check
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    cargo clippy --locked --all-targets -- -D warnings
# The NEON module is not compiled for x86_64, so lint the aarch64 target too.
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    cargo clippy --locked --all-targets --target aarch64-unknown-linux-gnu -- -D warnings
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    cargo test --locked
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    ISYNTAX_SIMD=scalar cargo test --locked
# The x86-64-v4 build (docs/performance.md, the bench's v4 column) must give the same bytes as the
# default one: the dsp unit tests, which hold every kernel against scalar, released, with their own
# cache mount, since RUSTFLAGS changes every artifact. Skipped on a runner without AVX-512.
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target-v4,target=/src/target \
    if grep -q avx512bw /proc/cpuinfo; then \
      RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" \
      cargo test --locked --release --lib dsp::; \
    else echo "no AVX-512: x86-64-v4 test skipped"; fi
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    ISYNTAX_SIMD=avx2 cargo test --locked
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    cargo build --locked --release --target aarch64-unknown-linux-gnu

# aarch64 tests, native on ubuntu-24.04-arm, both SIMD tiers (NEON, then ISYNTAX_SIMD=scalar): only
# the tests that the target changes. The `dsp::` unit tests hold the NEON kernels against scalar,
# the tier detection and the other dsp checks; the rest of the unit tests (the decoder, the cache,
# the container) do not depend on the target. The full suite runs on x86_64 in `ci`; Go parity on
# aarch64 is the parity-arm64 stage.
FROM src AS arm64
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    cargo test --locked --lib dsp::
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    ISYNTAX_SIMD=scalar cargo test --locked --lib dsp::

# Go parity: the Go oracle writes the inputs and decodes them; the Rust example decodes the same
# inputs; the two output trees must be equal. The oracle imports isyntax-go's internal fixture
# package, from the snapshot in tests/parity/go/upstream.
FROM golang:1.27-bookworm AS go-oracle
ENV GOTOOLCHAIN=local CGO_ENABLED=0
WORKDIR /src
COPY tests/parity/go tests/parity/go
RUN --mount=type=cache,id=isyntax-go-build,target=/root/.cache/go-build cd tests/parity/go/oracle && go build -o /out/oracle .
# The same oracle built for x86-64-v4 (the bench's v4 column); the parity stages use /out/oracle.
RUN --mount=type=cache,id=isyntax-go-build,target=/root/.cache/go-build cd tests/parity/go/oracle && GOAMD64=v4 go build -o /out/oracle-v4 .

FROM go-oracle AS go-run
RUN /out/oracle -gen /inputs && ls /inputs | wc -l && du -sh /inputs
RUN /out/oracle /inputs /go-out && ls /go-out | wc -l && du -sh /go-out
# The subset the scalar and AVX2 tiers and the aarch64 oracle run: every input
# that is not a mutation (`~` in its name), and every 16th mutation.
RUN mkdir /sub && ls /inputs | awk '!/~/ || NR % 16 == 0' | (cd /inputs && xargs cp -t /sub) \
    && /out/oracle /sub /go-sub && ls /sub | wc -l && ls /go-sub | wc -l

# The Go trees are bind mounts from go-run, not COPY: a COPY of the trees after `COPY . .` would
# run on every commit. The Rust output is deleted in the same RUN, so the layer (and the cache
# export) stays empty.
# `rw` on the go-run mounts: BuildKit keys a read-only mount by a hash of its whole content, a rw
# mount by go-run's own cache key (the writes are thrown away after the RUN).
FROM src AS parity
# The floors keep the diff from passing on an empty tree: most outputs have a line per operation.
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    --mount=type=bind,from=go-run,source=/inputs,target=/inputs,rw \
    --mount=type=bind,from=go-run,source=/go-out,target=/go-out,rw \
    --mount=type=bind,from=go-run,source=/sub,target=/sub,rw \
    --mount=type=bind,from=go-run,source=/go-sub,target=/go-sub,rw \
    cargo run --locked --release --example oracle -- /inputs /rust-out \
    && echo "$(ls /inputs | wc -l) inputs, $(ls /go-out | wc -l) outputs, $(cat /go-out/* | grep -c ' ok ') ok lines" \
    && [ "$(ls /go-out | wc -l)" -ge 500 ] && [ "$(cat /go-out/* | grep -c ' ok ')" -ge 3000 ] \
    && { diff -r /go-out /rust-out >/tmp/diff.txt || { head -80 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && ISYNTAX_ORACLE_THREADS=1 cargo run --locked --release --example oracle -- /inputs /rust-out-1t \
    && { diff -r /go-out /rust-out-1t >/tmp/diff.txt || { head -80 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && ISYNTAX_SIMD=scalar cargo run --locked --release --example oracle -- /sub /rust-out-scalar \
    && { diff -r /go-sub /rust-out-scalar >/tmp/diff.txt || { head -80 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && ISYNTAX_SIMD=avx2 cargo run --locked --release --example oracle -- /sub /rust-out-avx2 \
    && { diff -r /go-sub /rust-out-avx2 >/tmp/diff.txt || { head -80 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && rm -rf /rust-out /rust-out-1t /rust-out-scalar /rust-out-avx2

# aarch64 against Go, native on ubuntu-24.04-arm: the oracle example (the NEON kernels) on the
# subset, once per SIMD tier; the output must equal Go's.
FROM src AS parity-arm64
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    --mount=type=bind,from=go-run,source=/sub,target=/sub,rw \
    --mount=type=bind,from=go-run,source=/go-sub,target=/go-sub,rw \
    cargo run --locked --release --example oracle -- /sub /arm-out \
    && echo "$(ls /sub | wc -l) inputs, $(ls /go-sub | wc -l) outputs, $(cat /go-sub/* | grep -c ' ok ') ok lines" \
    && [ "$(ls /go-sub | wc -l)" -ge 100 ] && [ "$(cat /go-sub/* | grep -c ' ok ')" -ge 1000 ] \
    && { diff -r /go-sub /arm-out >/tmp/diff.txt || { head -80 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && ISYNTAX_SIMD=scalar cargo run --locked --release --example oracle -- /sub /arm-out-scalar \
    && { diff -r /go-sub /arm-out-scalar >/tmp/diff.txt || { head -80 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && rm -rf /arm-out /arm-out-scalar

# Fuzz seeds, from the Go fuzz tests' f.Add calls (the Go repo has no testdata/fuzz).
FROM go-oracle AS go-seeds
RUN /out/oracle -genseeds /seeds && find /seeds -type f | wc -l

# Fuzzing (fuzz.yml): nightly and cargo-fuzz, with the targets built into the image. The workflow
# runs it with `docker run`, so a BuildKit cache hit cannot skip the fuzzing. The same stage builds
# natively, with ASan, on x86_64 and on aarch64.
FROM toolchain AS fuzz-tools
RUN rustup toolchain install nightly --profile minimal \
    && cargo install cargo-fuzz --locked --version 0.13.1
# The fuzz crate's dependencies (its own lockfile), fetched before the source is copied; the stubs
# are the manifest's targets.
COPY tests/fuzz/Cargo.toml tests/fuzz/Cargo.lock tests/fuzz/
RUN mkdir src tests/fuzz/fuzz_targets && touch src/lib.rs \
    && for t in open decode hulsken tables kernels; do touch tests/fuzz/fuzz_targets/$t.rs; done \
    && cargo +nightly fetch --locked --manifest-path tests/fuzz/Cargo.toml \
    && rm -r src tests

FROM fuzz-tools AS fuzz
COPY . .
COPY --from=go-seeds /seeds /src/tests/fuzz/seeds
RUN cargo +nightly fuzz build --fuzz-dir tests/fuzz

# Benchmark (bench.yml): the Go oracle and the Rust bench and kernels examples in one image, the
# bench built twice: the default build, and x86-64-v4 (GOAMD64=v4, RUSTFLAGS -C target-cpu=x86-64-v4
# and the Zen 4/5 false-dependency features, docs/performance.md) as oracle-v4 and bench-v4;
# tests/parity/bench.sh prints both. `kernels` times each SIMD kernel on every tier of the CPU.
FROM src AS bench
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
COPY --from=go-oracle /out/oracle-v4 /usr/local/bin/oracle-v4
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    cargo build --locked --release --example bench --example kernels \
    && cp target/release/examples/bench /usr/local/bin/bench \
    && cp target/release/examples/kernels /usr/local/bin/kernels
# Its own cache mount (RUSTFLAGS changes every artifact); the sources are touched first, since the
# mount is shared by every branch. `--target` keeps RUSTFLAGS off the build scripts, which run
# here on a runner that may have no AVX-512.
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target-v4,target=/src/target \
    find src examples -name '*.rs' -exec touch -d '2100-01-01' {} + \
    && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --locked --release --target x86_64-unknown-linux-gnu --example bench \
    && cp target/x86_64-unknown-linux-gnu/release/examples/bench /usr/local/bin/bench-v4

# PGO and BOLT (pgo.yml, dispatch only; docs/performance.md): cargo-pgo builds the bench example
# instrumented, trains it on the fixtures, rebuilds it with the profile, and, when BOLT is
# installed, once more through BOLT. The tools are installed before `COPY . .`, so a new commit
# does not install them again. BOLT comes from apt.llvm.org (bookworm has none of its own); if the
# repository or the package is missing the image still builds and the job reports PGO only.
FROM toolchain AS pgo-tools
RUN (rustup component add llvm-tools || rustup component add llvm-tools-preview) \
    && cargo install cargo-pgo --locked
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg \
    && { wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
         && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-19 main" > /etc/apt/sources.list.d/llvm.list \
         && apt-get update && apt-get install -y --no-install-recommends bolt-19 libbolt-19-dev \
         && ln -s /usr/lib/llvm-19/bin/llvm-bolt /usr/local/bin/llvm-bolt \
         && ln -s /usr/lib/llvm-19/bin/merge-fdata /usr/local/bin/merge-fdata \
         && ln -s "$(find /usr/lib/llvm-19 -name libbolt_rt_instr.a | head -n 1)" /usr/local/lib/libbolt_rt_instr.a \
         && test -s /usr/local/lib/libbolt_rt_instr.a; } \
    || echo "BOLT is not available: the PGO job reports PGO only"; \
    rm -rf /var/lib/apt/lists/*

# bench (default build), bench-pgo and, when BOLT worked, bench-bolt (PGO then BOLT), all in
# /usr/local/bin with the Go oracle; tests/parity/pgo.sh times them. There is no tests/data: the
# fixtures are the Go oracle's (`oracle -genbench`), and the training runs are `bench <fixtures> 5 1`
# (one thread) and `bench <fixtures> 5` (default threads). The profiles of an earlier run stay in the
# cache mount, so they are deleted first.
FROM pgo-tools AS pgo
COPY . .
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target-pgo,target=/src/target \
    find src examples -name '*.rs' -exec touch -d '2100-01-01' {} + && touch -d '2100-01-01' Cargo.toml \
    && rm -rf target/pgo-profiles target/bolt-profiles && find target -name 'bench-bolt-*' -delete \
    && oracle -genbench /tmp/fixtures \
    && host=$(rustc -vV | sed -n 's/^host: //p') && bin=target/$host/release/examples/bench \
    && cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench \
    && cargo pgo build -- --locked --example bench \
    && cargo pgo run -- --locked --example bench -- /tmp/fixtures 5 1 \
    && cargo pgo run -- --locked --example bench -- /tmp/fixtures 5 \
    && cargo pgo optimize build -- --locked --example bench && cp $bin /usr/local/bin/bench-pgo \
    && if command -v llvm-bolt >/dev/null && command -v merge-fdata >/dev/null; then \
         { cargo pgo bolt build --with-pgo -- --locked --example bench \
           && "$(dirname $bin)/bench-bolt-instrumented" /tmp/fixtures 5 1 >/dev/null \
           && cargo pgo bolt optimize --with-pgo -- --locked --example bench \
           && cp "$(find target -type f -name 'bench-bolt-optimized' | head -n 1)" /usr/local/bin/bench-bolt; } \
         || echo "BOLT failed: the job reports PGO only"; \
       else echo "no BOLT: the job reports PGO only"; fi

# Quality gate (quality.yml). The OpenSlide set has no iSyntax file, so the real files are the parity
# corpus: the six bench fixtures and the 24 largest inputs of `oracle -gen` that are not mutants
# (docs/parity.md). The Go oracle (-ppm) and examples/ppm.rs decode every tile of every level of each;
# tests/parity/quality.sh diffs the two sets of PPM files byte for byte and floors the tiles compared.

# The input set (/in) and the Go side of the gate (/go-out), as go-run does for parity; the Rust
# stage bind-mounts both. The set is the bench fixtures and the 24 largest files of `oracle -gen`
# without `~` in the name (not a mutant) that are not .blk, .b64 or .ct.
FROM go-oracle AS go-quality
RUN /out/oracle -genbench /in && /out/oracle -gen /gen \
    && (cd /gen && ls -S | grep -v '~' | grep -Ev '[.](blk|b64|ct)$' | head -n 24 | xargs cp -t /in) \
    && echo "$(ls /in | wc -l) inputs" && du -sh /in \
    && /out/oracle -ppm /in /go-out 2>/go-ppm.err && { tail -n 20 /go-ppm.err; echo "$(wc -l </go-ppm.err) lines on stderr"; } \
    && ls /go-out | wc -l && du -sh /go-out

# One RUN: the Rust side decodes into a scratch dir that quality.sh deletes, so the layer holds only
# /report. `rw` on the go-quality mounts, as in `parity`.
FROM src AS quality
RUN --mount=type=cache,id=isyntax-cargo-registry,target=/usr/local/cargo/registry --mount=type=cache,id=isyntax-target,target=/src/target \
    --mount=type=bind,from=go-quality,source=/in,target=/in,rw \
    --mount=type=bind,from=go-quality,source=/go-out,target=/go-out,rw \
    cargo build --locked --release --example ppm \
    && mkdir -p /report \
    && sh tests/parity/quality.sh /in /go-out /report

# The report as build output: docker/build-push-action with `outputs: type=local,dest=quality-out`.
FROM scratch AS quality-report
COPY --from=quality /report /
