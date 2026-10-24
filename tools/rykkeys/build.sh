#!/data/data/com.termux/files/usr/bin/sh
# Build the headset helper: RykKeys.java -> $OUT/keys.jar, the way `td install`
# builds TdAlarm (ecj + dx; the Termux ecj wrapper already has android.jar).
#
#   tools/rykkeys/build.sh [OUTDIR]     (default ~/.cache/ryk-dev)
#
# Optional: ryk --serve builds the same jar from the source embedded in the binary on first
# use. This script is for working on RykKeys.java: it builds beside, not over, ryk's own jar,
# and RYK_KEYS_JAR=~/.cache/ryk-dev/keys.jar makes ryk use (and never rebuild) the result.
# chain-test.py builds the same way.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-${XDG_CACHE_HOME:-$HOME/.cache}/ryk-dev}
mkdir -p "$OUT/classes"
rm -f "$OUT"/classes/*.class
ecj -nowarn -d "$OUT/classes" "$HERE/RykKeys.java"
dx --dex --output="$OUT/keys.jar" "$OUT/classes"
echo "built $OUT/keys.jar (RYK_KEYS_JAR=$OUT/keys.jar to use it)"
