#!/bin/zsh
# Assemble the app-icon-enabled macOS simulator from a fresh source build:
# backfill StreamingAssets (field scenes) from a published package, drop the
# icon artwork next to PlayerIcon.icns, inject the dual-appearance Assets.car
# and re-seal ad-hoc. Run AFTER IconBuild.Mac produced
# sdsim/Builds/DonkeySimMac/donkey_sim.app.
#
# usage: tools/simulator-icon/assemble_source_build.sh <donor-donkey_sim.app>
set -euo pipefail
cd "$(dirname "$0")/../.."

DONOR="${1:?usage: assemble_source_build.sh <donor donkey_sim.app with StreamingAssets>}"
APP="sdsim/Builds/DonkeySimMac/donkey_sim.app"
[[ -d "$APP/Contents/MacOS" ]] || { echo "build output missing: $APP" >&2; exit 1; }
[[ -d "$DONOR/Contents/Resources/Data/StreamingAssets" ]] || { echo "donor has no StreamingAssets: $DONOR" >&2; exit 1; }

# 场景/字段资产：源码工程不带 AssetBundles，从已发布包回填
rsync -a "$DONOR/Contents/Resources/Data/StreamingAssets/" "$APP/Contents/Resources/Data/StreamingAssets/"

for name in icon-day.png icon-night.png; do
  cp "tools/simulator-icon/AppIcon.icon/Assets/$name" "$APP/Contents/Resources/$name"
done

zsh tools/simulator-icon/apply-dark-icon.sh "$APP"

# 注入改动了 Resources——统一 ad-hoc 重封（未签名包 macOS 不理会外观目录）
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
echo "assembled: $APP"
