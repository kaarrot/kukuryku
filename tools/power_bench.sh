#!/usr/bin/env bash
# One ryk utterance with ~1 Hz timestamped samples of temps, freqs, and battery.
#
# Refuses to start when the gold thermal zones (cpu-1-*-usr) are already above
# RYK_COOL_C (default 58), unless RYK_BENCH_FORCE=1. A plugged-in battery
# (Charging / Full / Not charging) makes the joule columns meaningless — the
# script warns and still prints the row; cpu_s (process CPU seconds) is the
# load number to compare then. Never calls dumpsys (it hangs on this phone).
#
# Usage: tools/power_bench.sh [config-name]
#   RYK_BIN           binary (default target/release/ryk)
#   RYK_BENCH_TEXT    utterance; else RYK_BENCH_FILE; else a short lighthouse paragraph
#   RYK_COOL_C        gold-zone refuse threshold in °C (default 58)
#   RYK_BENCH_FORCE   set to 1 to start even when the golds are already hot
#   RYK_BENCH_IDLE_S  seconds of idle sampling before the run, for the net
#                     columns (default 5; 0 skips it)
#   RYK_BENCH_KEEP    directory to copy the samples and ryk's stderr into
#   RYK_CURRENT_UNIT  ua|ma to override the battery current unit detection
#
# Output columns: wall_rtf/gap_s from ryk's `[kokoro] done:` line (speak clock,
# so model compile is not in the ratio); cpu_s = user+sys CPU of ryk and the
# children it waited for (pacat; not the detached PulseAudio daemon); peak_c /
# mean_c = hottest gold zone; mean_w = battery power over the run; j_per_audio_s
# = run energy / audio seconds; net_* = the change from the idle baseline, from
# signed currents, so it holds for either sign convention. Unplugged, net_* is
# the load. Charging, only net_* means anything, and only while the charger
# holds a constant current (Samsung lowers it when the phone is hot). Run
# unplugged, starting cool, or the watts are not a load number.
#
# The sampler uses only shell builtins (plus one `sleep` per sample): forking
# `cat` per thermal zone made the old "1 Hz" loop take ~2 s per sample under
# load and add its own CPU and heat to the numbers. Energy is integrated over
# the real sample timestamps, not an assumed 1 s period.

set -u
# EPOCHREALTIME and awk both follow LC_NUMERIC; keep the decimal point a dot.
export LC_NUMERIC=C

cfg="${1:-default}"
bin="${RYK_BIN:-target/release/ryk}"
cool_c="${RYK_COOL_C:-58}"
force="${RYK_BENCH_FORCE:-0}"
idle_s="${RYK_BENCH_IDLE_S:-5}"
bat=/sys/class/power_supply/battery

if [[ -n "${RYK_BENCH_FILE:-}" ]]; then
    text=$(<"$RYK_BENCH_FILE")
elif [[ -n "${RYK_BENCH_TEXT:-}" ]]; then
    text="$RYK_BENCH_TEXT"
else
    text="She stood at the lighthouse and watched the boats come in. The wind was cold and the light turned slowly. A child on the pier waved once and then ran back to the road. By evening the fog had covered the rocks and the bell started up. They stayed until the last boat was only a sound."
fi

# Resolve the zone files once. Per-core `*-usr` zones are the live readings
# (lmh-dcvs-* is a static 75 °C sentinel on SD855 Samsung firmware).
little_zones=()
gold_zones=()
for z in /sys/class/thermal/thermal_zone*; do
    read -r typ <"$z/type" 2>/dev/null || continue
    [[ -r "$z/temp" ]] || continue
    case "$typ" in
        cpu-0-*-usr) little_zones+=("$z/temp") ;;
        cpu-1-*-usr) gold_zones+=("$z/temp") ;;
    esac
done

# Hottest zone of the given files in °C, into $MAXC (0 if none readable).
max_c() {
    local f v m=0
    for f in "$@"; do
        read -r v <"$f" 2>/dev/null || continue
        [[ "$v" =~ ^-?[0-9]+$ ]] || continue
        v=$((v / 1000))
        ((v > m)) && m=$v
    done
    MAXC=$m
}

# File contents into $VAL, 0 when unreadable or not an integer.
read_int() {
    VAL=0
    read -r VAL <"$1" 2>/dev/null || VAL=0
    [[ "$VAL" =~ ^-?[0-9]+$ ]] || VAL=0
}

# Battery current in µA. batt_current_ua_now is Samsung's explicit-µA file.
# current_now is µA per the power_supply ABI, but Samsung reports mA there
# (e.g. 399 next to voltage_now=3890000 µV), which made the old watts 1000x
# too small. A |value| under 20000 cannot be µA on a running phone (20 mA),
# while 20000 mA would be 20 A, so small values are taken as mA.
cur_file="$bat/current_now"
[[ -r "$bat/batt_current_ua_now" ]] && cur_file="$bat/batt_current_ua_now"
read_ua() {
    read_int "$cur_file"
    local v=$VAL a=${VAL#-}
    case "${RYK_CURRENT_UNIT:-}" in
        ua) ;;
        ma) v=$((v * 1000)) ;;
        *) if [[ "$cur_file" == */current_now ]] && ((a < 20000)); then v=$((v * 1000)); fi ;;
    esac
    UA=$v
}

# One sample line: t_s little_c gold_c f0 f4 f7 ua uv
sample() {
    local t=$EPOCHREALTIME c0 c1 f0 f4 f7 uv
    max_c "${little_zones[@]}"; c0=$MAXC
    max_c "${gold_zones[@]}"; c1=$MAXC
    read_int /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; f0=$VAL
    read_int /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq; f4=$VAL
    read_int /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq; f7=$VAL
    read_ua
    read_int "$bat/voltage_now"; uv=$VAL
    printf '%s %s %s %s %s %s %s %s\n' "$t" "$c0" "$c1" "$f0" "$f4" "$f7" "$UA" "$uv"
}

max_c "${gold_zones[@]}"
gold=$MAXC
if ((${#gold_zones[@]} > 0 && gold > cool_c)) && [[ "$force" != "1" ]]; then
    echo "power_bench: gold zones at ${gold}°C (cpu-1-*-usr), above ${cool_c}°C. Let the phone cool, or set RYK_BENCH_FORCE=1." >&2
    exit 2
fi

status=unknown
read -r status <"$bat/status" 2>/dev/null || status=unknown
# "Not charging" is still plugged in: the charger carries the load and the
# battery current stays near zero.
case "$status" in
    [Cc]harging|[Ff]ull|[Nn]ot\ [Cc]harging)
        echo "power_bench: battery status is ${status}. Joules are not a load number — unplug and rerun (compare cpu_s meanwhile)." >&2
        ;;
esac

if [[ ! -x "$bin" && ! -f "$bin" ]]; then
    echo "power_bench: binary not found: $bin" >&2
    exit 1
fi

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
samples="$workdir/samples"
idle="$workdir/idle"
err="$workdir/err"
runflag="$workdir/run"
: >"$samples"
: >"$idle"

if ((idle_s > 0)); then
    for ((i = 0; i < idle_s; i++)); do
        sample >>"$idle"
        sleep 1
    done
    sample >>"$idle"
fi

: >"$runflag"
(
    while [[ -f "$runflag" ]]; do
        sample >>"$samples"
        sleep 1
    done
    sample >>"$samples"
) &
sampid=$!

export KOKORO_VERBOSE=1
TIMEFORMAT='%3R %3U %3S'
{ time "$bin" "$text" >"$workdir/out" 2>"$err"; } 2>"$workdir/time"
rc=$?
rm -f "$runflag"
wait "$sampid" 2>/dev/null || true

read -r real_s user_s sys_s <"$workdir/time" || true
cpu_s=$(awk -v u="${user_s:-0}" -v s="${sys_s:-0}" 'BEGIN { printf "%.2f", u + s }')

done_line=$(grep '\[kokoro\] done:' "$err" | tail -n 1 || true)
wall_rtf=$(printf '%s\n' "$done_line" | sed -n 's/.*wall RTF \([0-9.][0-9.]*\).*/\1/p')
gap_s=$(printf '%s\n' "$done_line" | sed -n 's/.*gap \([0-9.][0-9.]*\)s.*/\1/p')
audio_s=$(printf '%s\n' "$done_line" | sed -n 's/.*done: \([0-9.][0-9.]*\)s audio.*/\1/p')
wall_rtf=${wall_rtf:-na}
gap_s=${gap_s:-na}
audio_s=${audio_s:-0}

# Battery power of a sample file, trapezoid over the real timestamps.
# Prints: mean_abs_w mean_signed_w seconds peak_gold_c mean_gold_c nsamples
integrate='
    NF < 8 { next }
    {
        p = $7 * $8 / 1e12
        if (n > 0) { dt = $1 - t; e += (p + pp) / 2 * dt; T += dt }
        t = $1; pp = p; n++
        if ($3 > peak) peak = $3
        gsum += $3
    }
    END {
        if (n < 2 || T <= 0) { printf "0 0 0 %d %d %d\n", peak, (n ? gsum / n : 0), n; exit }
        w = e / T
        printf "%.3f %.3f %.1f %d %.0f %d\n", (w < 0 ? -w : w), w, T, peak, gsum / n, n
    }'
read -r mean_w run_signed_w run_t peak mean_c nsample < <(awk "$integrate" "$samples")
read -r idle_w idle_signed_w _ _ _ _ < <(awk "$integrate" "$idle")

j_per="na"
net_w="na"
net_j_per="na"
if awk "BEGIN { exit !($audio_s > 0) }"; then
    j_per=$(awk -v w="$mean_w" -v t="$run_t" -v a="$audio_s" 'BEGIN { printf "%.3f", w * t / a }')
    if ((idle_s > 0)); then
        net_w=$(awk -v r="$run_signed_w" -v i="$idle_signed_w" \
            'BEGIN { d = i - r; printf "%.3f", (d < 0 ? -d : d) }')
        net_j_per=$(awk -v w="$net_w" -v t="$run_t" -v a="$audio_s" 'BEGIN { printf "%.3f", w * t / a }')
    fi
fi

printf 'config=%s wall_rtf=%s gap_s=%s cpu_s=%s real_s=%s peak_c=%s mean_c=%s mean_w=%s idle_w=%s net_w=%s j_per_audio_s=%s net_j_per_audio_s=%s samples=%s battery=%s rc=%s\n' \
    "$cfg" "$wall_rtf" "$gap_s" "$cpu_s" "${real_s:-na}" "$peak" "$mean_c" "$mean_w" "${idle_w:-na}" "$net_w" "$j_per" "$net_j_per" "$nsample" "$status" "$rc"

if [[ -n "${RYK_BENCH_KEEP:-}" ]]; then
    mkdir -p "$RYK_BENCH_KEEP"
    cp "$samples" "$RYK_BENCH_KEEP/$cfg.samples"
    cp "$idle" "$RYK_BENCH_KEEP/$cfg.idle"
    cp "$err" "$RYK_BENCH_KEEP/$cfg.err"
fi
if [[ -z "$done_line" ]]; then
    echo "power_bench: no [kokoro] done: line (set RYK_BENCH_KEEP=dir to keep ryk's stderr)" >&2
    tail -n 30 "$err" >&2 || true
fi
exit "$rc"
