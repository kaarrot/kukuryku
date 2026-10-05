#!/usr/bin/env bash
# One ryk utterance with 1 Hz samples of temps, freqs, and battery.
#
# Refuses to start when the gold thermal zones (cpu-1-*) are already above
# RYK_COOL_C (default 58), unless RYK_BENCH_FORCE=1. A Charging battery makes
# the joule column meaningless — the script warns and still prints the row.
# Never calls dumpsys (it hangs on this phone).
#
# Usage: tools/power_bench.sh [config-name]
#   RYK_BIN          binary (default target/release/ryk)
#   RYK_BENCH_TEXT   utterance; else RYK_BENCH_FILE; else a short lighthouse paragraph
#   RYK_COOL_C       gold-zone refuse threshold in °C (default 58)
#   RYK_BENCH_FORCE  set to 1 to start even when the golds are already hot
#
# The summary's wall RTF comes from ryk's `[kokoro] done:` line (speak clock,
# not the process total, so model compile is not in the ratio). Run unplugged,
# starting cool, or the watts are not a load number.

set -u

cfg="${1:-default}"
bin="${RYK_BIN:-target/release/ryk}"
cool_c="${RYK_COOL_C:-58}"
force="${RYK_BENCH_FORCE:-0}"

if [[ -n "${RYK_BENCH_FILE:-}" ]]; then
    text=$(cat "$RYK_BENCH_FILE")
elif [[ -n "${RYK_BENCH_TEXT:-}" ]]; then
    text="$RYK_BENCH_TEXT"
else
    text="She stood at the lighthouse and watched the boats come in. The wind was cold and the light turned slowly. A child on the pier waved once and then ran back to the road. By evening the fog had covered the rocks and the bell started up. They stayed until the last boat was only a sound."
fi

# Hottest thermal zone whose type starts with PREFIX, in °C. Empty if none.
max_zone_c() {
    local prefix="$1" max=-1000 found=0 z typ milli c
    for z in /sys/class/thermal/thermal_zone*; do
        [[ -r "$z/type" && -r "$z/temp" ]] || continue
        typ=$(cat "$z/type" 2>/dev/null || true)
        case "$typ" in
            ${prefix}*) ;;
            *) continue ;;
        esac
        milli=$(cat "$z/temp" 2>/dev/null || true)
        [[ "$milli" =~ ^-?[0-9]+$ ]] || continue
        c=$((milli / 1000))
        found=1
        if (( c > max )); then max=$c; fi
    done
    if (( found )); then printf '%s\n' "$max"; fi
}

gold=$(max_zone_c "cpu-1-" || true)
if [[ -n "$gold" && "$gold" -gt "$cool_c" && "$force" != "1" ]]; then
    echo "power_bench: gold zones at ${gold}°C (cpu-1-*), above ${cool_c}°C. Let the phone cool, or set RYK_BENCH_FORCE=1." >&2
    exit 2
fi

status=$(cat /sys/class/power_supply/battery/status 2>/dev/null || echo unknown)
case "$status" in
    [Cc]harging|[Ff]ull)
        echo "power_bench: battery status is ${status}. Joules are not a load number — unplug and rerun." >&2
        ;;
esac

if [[ ! -x "$bin" && ! -f "$bin" ]]; then
    echo "power_bench: binary not found: $bin" >&2
    exit 1
fi

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
samples="$workdir/samples"
err="$workdir/err"
runflag="$workdir/run"
: >"$samples"
: >"$runflag"

read_khz() {
    local f
    f=$(cat "$1" 2>/dev/null || echo 0)
    [[ "$f" =~ ^[0-9]+$ ]] || f=0
    printf '%s\n' "$f"
}

(
    while [[ -f "$runflag" ]]; do
        c0=$(max_zone_c "cpu-0-" || true); c0=${c0:-0}
        c1=$(max_zone_c "cpu-1-" || true); c1=${c1:-0}
        lmh=$(max_zone_c "lmh-dcvs" || true); lmh=${lmh:-0}
        f0=$(read_khz /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
        f4=$(read_khz /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq)
        f7=$(read_khz /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq)
        ua=$(cat /sys/class/power_supply/battery/current_now 2>/dev/null || echo 0)
        uv=$(cat /sys/class/power_supply/battery/voltage_now 2>/dev/null || echo 0)
        [[ "$ua" =~ ^-?[0-9]+$ ]] || ua=0
        [[ "$uv" =~ ^-?[0-9]+$ ]] || uv=0
        printf '%s %s %s %s %s %s %s %s\n' "$c0" "$c1" "$lmh" "$f0" "$f4" "$f7" "$ua" "$uv" >>"$samples"
        sleep 1
    done
) &
sampid=$!

export KOKORO_VERBOSE=1
set +e
"$bin" "$text" >"$workdir/out" 2>"$err"
rc=$?
set -e
rm -f "$runflag"
wait "$sampid" 2>/dev/null || true

done_line=$(grep '\[kokoro\] done:' "$err" | tail -n 1 || true)
wall_rtf=$(printf '%s\n' "$done_line" | sed -n 's/.*wall RTF \([0-9.][0-9.]*\).*/\1/p')
gap_s=$(printf '%s\n' "$done_line" | sed -n 's/.*gap \([0-9.][0-9.]*\)s.*/\1/p')
audio_s=$(printf '%s\n' "$done_line" | sed -n 's/.*done: \([0-9.][0-9.]*\)s audio.*/\1/p')
wall_rtf=${wall_rtf:-na}
gap_s=${gap_s:-na}
audio_s=${audio_s:-0}

# columns: c0 c1 lmh f0 f4 f7 ua(µA) uv(µV)
read -r peak mean_w nsample <<EOF
$(awk '
    NF < 8 { next }
    {
        n++
        for (i = 1; i <= 3; i++) if ($i > peak) peak = $i
        ua = $7; if (ua < 0) ua = -ua
        uv = $8; if (uv < 0) uv = -uv
        watts += (ua * uv) / 1e12
    }
    END {
        if (n < 1) { print "0 0 0"; exit }
        printf "%d %.3f %d\n", peak, watts / n, n
    }
' "$samples")
EOF
peak=${peak:-0}
mean_w=${mean_w:-0}
nsample=${nsample:-0}

j_per=""
if awk "BEGIN { exit !($audio_s > 0) }"; then
    # samples are 1 Hz, so nsample seconds is the integration window.
    j_per=$(awk -v w="$mean_w" -v n="$nsample" -v a="$audio_s" 'BEGIN { printf "%.3f", (w * n) / a }')
else
    j_per="na"
fi

printf 'config=%s wall_rtf=%s gap_s=%s peak_c=%s mean_w=%s j_per_audio_s=%s samples=%s battery=%s rc=%s\n' \
    "$cfg" "$wall_rtf" "$gap_s" "$peak" "$mean_w" "$j_per" "$nsample" "$status" "$rc"
if [[ -z "$done_line" ]]; then
    echo "power_bench: no [kokoro] done: line (see $err before it is removed — rerun with the log kept if this persists)" >&2
    tail -n 30 "$err" >&2 || true
fi
exit "$rc"
