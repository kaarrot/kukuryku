#!/data/data/com.termux/files/usr/bin/sh
# Build the headset helper: RykKeys.java -> $OUT/keys.jar, the way `td install`
# builds TdAlarm (ecj + dx; the Termux ecj wrapper already has android.jar).
#
#   tools/rykkeys/build.sh [OUTDIR]     (default ~/.cache/ryk, where ryk --serve looks)
#
# ryk --serve starts the helper by itself once the jar exists. Set
# RYK_HEADSET=0 to keep it off.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-${XDG_CACHE_HOME:-$HOME/.cache}/ryk}
mkdir -p "$OUT/classes"
rm -f "$OUT"/classes/*.class
ecj -nowarn -d "$OUT/classes" "$HERE/RykKeys.java"
dx --dex --output="$OUT/keys.jar" "$OUT/classes"
echo "built $OUT/keys.jar"
