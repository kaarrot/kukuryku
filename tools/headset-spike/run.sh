#!/data/data/com.termux/files/usr/bin/sh
# Throwaway spike for docs/headset-control.md. Does a MediaSession owned by
# Termux's uid receive this headset's buttons while Termux is playing audio?
#
#   tools/headset-spike/run.sh [SECS] [handoff]     (SECS default 60)
#
# With `guided` it prompts for one gesture at a time (double tap twice, triple tap, single tap,
# swipes, swipe-and-hold; about 65 s) and prints which key each
# produced, so you do not have to remember the order.
#
# With `handoff` the first key you press releases the session and is
# re-dispatched to whoever had the buttons before (Apple Music): it tests
# whether ryk can give a key back.
#
# Press the headset buttons; every key event is printed with a ms timestamp.
# Builds with ecj + dx (the way `td install` builds TdAlarm) and runs under
# app_process as the Termux uid. A quiet tone plays meanwhile, so Termux is
# the app that "last played audio", as it is while ryk speaks.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${TMPDIR:-/data/data/com.termux/files/usr/tmp}/ryk-keys-spike
SECS=${1:-60}
MODE=${2:-}
[ "$MODE" = guided ] && SECS=75
mkdir -p "$OUT/classes"
ecj -nowarn -d "$OUT/classes" "$HERE/MediaSpike.java"
dx --dex --output="$OUT/spike.jar" "$OUT/classes"

pulseaudio --check 2>/dev/null || pulseaudio --start --exit-idle-time=30
sleep 1
python3 -I -c '
import math, struct, sys
period = b"".join(struct.pack("<f", 0.002 * math.sin(2 * math.pi * 240 * i / 24000)) for i in range(100))
sys.stdout.buffer.write(period * (240 * '"$SECS"'))' |
    pacat --raw --format=float32le --rate 24000 --channels=1 --latency-msec=100 &
PACAT=$!
trap 'kill $PACAT 2>/dev/null' EXIT INT TERM

echo "session up for ${SECS}s: press headset buttons now (timestamps are ms mod 100000)"
env -u LD_PRELOAD -u LD_LIBRARY_PATH CLASSPATH="$OUT/spike.jar" \
    /system/bin/app_process -Xnoimage-dex2oat / MediaSpike session "$SECS" $MODE
