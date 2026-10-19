#!/bin/sh
set -eu

# Run each target for a bounded time and keep discoveries outside tracked seeds.
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fuzz_bin_dir=${1:-build/fuzz/fuzz}
fuzz_seconds=${2:-30}
case "$fuzz_seconds" in
    ''|*[!0-9]*|0) echo 'Duration must be a positive number of seconds' >&2; exit 2 ;;
esac
fuzz_bin_dir=$(CDPATH= cd -- "$fuzz_bin_dir" && pwd)
for name in core unicode codecs; do
    corpus="$fuzz_bin_dir/corpus/$name"
    artifacts="$fuzz_bin_dir/artifacts/$name"
    mkdir -p "$corpus" "$artifacts"
    cp "$script_dir"/seeds/* "$corpus/"
    "$fuzz_bin_dir/fuzz_$name" "$corpus" \
        -max_total_time="$fuzz_seconds" -max_len=4096 -timeout=10 \
        -rss_limit_mb=2048 -seed=1 -print_final_stats=1 \
        -artifact_prefix="$artifacts/"
done
