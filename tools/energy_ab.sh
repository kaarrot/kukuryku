#!/usr/bin/env bash
# Unattended alternating A/B energy runs around tools/power_bench.sh.
#
# Pair i runs arm A and arm B once each, AB on even pairs and BA on odd ones, and
# every run starts only after the gold zones have cooled to RYK_COOL_C. Each row
# keeps power_bench's key=value line plus pair/arm/position. With the same binary
# and environment on both arms (the default) this is an A/A null run: it measures
# the noise floor of the method, not a code change. Feed rows.log to
# tools/energy_ab_analyze.py for paired differences and the smallest effect the
# run could have seen.
#
# Run it unplugged, screen off, and leave the phone alone: the script holds a
# Termux wake lock so the CPU stays up, but anything else running on the phone
# (including a busy Claude Code session) lands in the idle floor and the noise.
#
# Usage: tools/energy_ab.sh
#   RYK_BIN_A / RYK_BIN_B   arm binaries (default target/release/ryk for both)
#   RYK_AB_ENV_A / _B       extra KEY=VALUE words for that arm, e.g.
#                           "KOKORO_TRACT_S1_DENSE_KERNEL=legacy"
#   RYK_AB_PAIRS            pairs to run (default 10)
#   RYK_AB_MAX_MIN          stop starting new pairs after this many minutes (60)
#   RYK_AB_OUT              output directory (default tools/bench_out/energy-ab-<stamp>)
#   RYK_AB_SINK             discard (default, silent) or real (ryk's own player)
#   RYK_AB_COOL_WAIT_S      longest wait for one cool-down before giving up (900)
#   RYK_AB_START_DELAY_S    wait this long before the first check, to unplug and
#                           turn the screen off after launching (default 0)
#   RYK_AB_ALLOW_PLUGGED    set to 1 to run while charging (joules are meaningless)
#   RYK_COOL_C              gold-zone start threshold in °C (default 52)
#   RYK_BENCH_IDLE_S        idle baseline seconds before each run (default 10)
#   RYK_BENCH_TEXT / _FILE  utterance, as for power_bench.sh
# Exit codes: 0 done, 3 charging, 4 cool-down timeout, 5 a run failed.

set -u
export LC_NUMERIC=C

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin_a="${RYK_BIN_A:-$root/target/release/ryk}"
bin_b="${RYK_BIN_B:-$bin_a}"
env_a="${RYK_AB_ENV_A:-}"
env_b="${RYK_AB_ENV_B:-}"
pairs="${RYK_AB_PAIRS:-10}"
max_min="${RYK_AB_MAX_MIN:-60}"
sink="${RYK_AB_SINK:-discard}"
cool_wait="${RYK_AB_COOL_WAIT_S:-900}"
out="${RYK_AB_OUT:-$root/tools/bench_out/energy-ab-$(date +%Y%m%d-%H%M)}"
bat=/sys/class/power_supply/battery

export RYK_COOL_C="${RYK_COOL_C:-52}"
export RYK_BENCH_IDLE_S="${RYK_BENCH_IDLE_S:-10}"
export KOKORO_GOVERNOR="${KOKORO_GOVERNOR:-off}"

for b in "$bin_a" "$bin_b"; do
    [[ -x "$b" ]] || { echo "energy_ab: binary not executable: $b" >&2; exit 1; }
done

mkdir -p "$out/raw"
rows="$out/rows.log"
events="$out/events.log"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$events" >&2; }

sink_path="$PATH"
if [[ "$sink" == discard ]]; then
    mkdir -p "$out/discard-sink"
    printf '#!%s\ncat >/dev/null\n' "$(command -v sh)" >"$out/discard-sink/ffplay"
    chmod 755 "$out/discard-sink/ffplay"
    sink_path="$out/discard-sink:$PATH"
fi

status() { local s=unknown; read -r s <"$bat/status" 2>/dev/null || true; printf '%s' "$s"; }

require_unplugged() {
    local s
    s=$(status)
    case "$s" in
        [Cc]harging | [Ff]ull | [Nn]ot\ [Cc]harging)
            if [[ "${RYK_AB_ALLOW_PLUGGED:-0}" != 1 ]]; then
                log "battery status is $s; unplug the phone (or RYK_AB_ALLOW_PLUGGED=1). Stopping."
                exit 3
            fi
            ;;
    esac
}

command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock
trap 'command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock' EXIT

((${RYK_AB_START_DELAY_S:-0} > 0)) && sleep "$RYK_AB_START_DELAY_S"

{
    echo "start=$(date -Is) pairs=$pairs max_min=$max_min sink=$sink cool_c=$RYK_COOL_C idle_s=$RYK_BENCH_IDLE_S"
    echo "bin_a=$bin_a sha256=$(sha256sum "$bin_a" | cut -d' ' -f1) env_a=[$env_a]"
    echo "bin_b=$bin_b sha256=$(sha256sum "$bin_b" | cut -d' ' -f1) env_b=[$env_b]"
    echo "governor=$KOKORO_GOVERNOR battery_pct=$(cat $bat/capacity 2>/dev/null) battery_temp_dC=$(cat $bat/temp 2>/dev/null)"
} | tee "$out/manifest.txt" >&2

require_unplugged

# One arm: wait out the cool-down, run power_bench once, append the row.
run_arm() {
    local pair=$1 arm=$2 pos=$3 bin extra waited=0 line rc errf
    if [[ "$arm" == A ]]; then bin=$bin_a; extra=$env_a; else bin=$bin_b; extra=$env_b; fi
    errf="$out/raw/p${pair}${arm}.stderr"
    while :; do
        require_unplugged
        # shellcheck disable=SC2086
        line=$(env PATH="$sink_path" $extra RYK_BIN="$bin" RYK_BENCH_KEEP="$out/raw" \
            bash "$root/tools/power_bench.sh" "p${pair}${arm}" 2>"$errf")
        rc=$?
        if ((rc == 2)) && grep -q 'gold zones at' "$errf"; then
            ((waited >= cool_wait)) && { log "pair $pair arm $arm: still hot after ${waited}s. Stopping."; exit 4; }
            sleep 20
            waited=$((waited + 20))
            continue
        fi
        break
    done
    printf 'pair=%s arm=%s pos=%s cooled_s=%s %s\n' "$pair" "$arm" "$pos" "$waited" "$line" >>"$rows"
    log "pair $pair $arm (pos $pos, cooled ${waited}s): $line"
    ((rc != 0)) && { log "run failed (rc=$rc); see $errf"; exit 5; }
    return 0
}

t0=$SECONDS
for ((p = 1; p <= pairs; p++)); do
    if ((p > 1 && (SECONDS - t0) >= max_min * 60)); then
        log "time budget of ${max_min} min reached after $((p - 1)) pairs"
        break
    fi
    if ((p % 2 == 1)); then order="A B"; else order="B A"; fi
    pos=1
    for arm in $order; do
        run_arm "$p" "$arm" "$pos"
        pos=$((pos + 1))
    done
done

log "done in $(((SECONDS - t0) / 60)) min; rows in $rows"
if command -v python3 >/dev/null 2>&1; then
    python3 "$root/tools/energy_ab_analyze.py" "$rows" | tee "$out/analysis.txt"
fi
