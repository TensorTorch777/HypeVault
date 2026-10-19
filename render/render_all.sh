#!/usr/bin/env bash
# Render every watch for the website.
#   render/render_all.sh preview   CPU, low samples — layout work while the GPU is busy
#   render/render_all.sh final     RTX GPU (OptiX), production quality
set -euo pipefail
MODE="${1:-final}"
BLENDER="${BLENDER:-$HOME/.local/opt/blender-4.5.14-linux-x64/blender}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../frontend/public/renders"
WATCHES="${WATCHES:-patek_nautilus ap_royal_oak lange_one vc_overseas rm_rm011}"

if [ "$MODE" = preview ]; then
  DEV=CPU; HERO_RES=900; HERO_S=64; FLAT_RES=700; FLAT_S=48; TT_RES=640; TT_S=24; TT_N=48; NICE="nice -n 15"
else
  DEV=GPU; HERO_RES=1800; HERO_S=512; FLAT_RES=1400; FLAT_S=384; TT_RES=1280; TT_S=256; TT_N=72; NICE=""
fi

for w in $WATCHES; do
  echo "== $w ($MODE)"
  $NICE "$BLENDER" -b -P "$HERE/scene.py" -- --watch "$w" --shot hero --out "$OUT/$w" --res $HERO_RES --samples $HERO_S --device $DEV | grep -E "\[render\]|Error" || true
  $NICE "$BLENDER" -b -P "$HERE/scene.py" -- --watch "$w" --shot flat --out "$OUT/$w" --res $FLAT_RES --samples $FLAT_S --device $DEV | grep -E "\[render\] done|Error" || true
  $NICE "$BLENDER" -b -P "$HERE/scene.py" -- --watch "$w" --shot turntable --frames $TT_N --out "$OUT/$w/turntable" --res $TT_RES --samples $TT_S --device $DEV | grep -E "\[render\] done|Error" || true
done

python3 - "$OUT" "$MODE" "$TT_N" <<'PY'
import json, sys, datetime
from pathlib import Path
out, mode, frames = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
manifest = {"quality": mode, "rendered_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", "watches": {}}
for d in sorted(p for p in out.iterdir() if p.is_dir() and p.name != "backdrops"):
    tt = sorted((d / "turntable").glob("f_*.webp"))
    manifest["watches"][d.name] = {"hero": (d / "hero.webp").exists(), "flat": (d / "flat.webp").exists(), "turntable_frames": len(tt)}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY
