#!/bin/bash
# DonkeySim macOS 分发打包脚本
# 用法: ./package_macos.sh <path/to/donkey_sim.app> [输出目录]
# 产出: DMG(带 Applications 快捷方式) + sha256 校验文件
set -e

APP="${1:?用法: package_macos.sh <donkey_sim.app 路径> [输出目录]}"
OUT_DIR="${2:-dist}"
STAGE="$OUT_DIR/stage"
VOLNAME="DonkeySim"

mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
# ad-hoc 重签名(深签),保证拷贝后签名有效
codesign --sign - --force --deep "$STAGE/$(basename "$APP")"
codesign --verify "$STAGE/$(basename "$APP")"

# 同目录有 README* 则一并放进 DMG(安装说明/Gatekeeper 绕过指引)
cp "$(dirname "$0")"/README*.md "$STAGE/" 2>/dev/null || true

# Applications 软链(拖动安装体验)
ln -sf /Applications "$STAGE/Applications"

VERSION="$(date +%Y%m%d)"
DMG="$OUT_DIR/DonkeySim-macOS-universal-$VERSION.dmg"
rm -f "$DMG"
hdiutil create -volname "$VOLNAME" -srcfolder "$STAGE" \
    -ov -format UDZO "$DMG" >/dev/null

shasum -a 256 "$DMG" | tee "$DMG.sha256"
echo "OK: $DMG ($(du -h "$DMG" | cut -f1))"
rm -rf "$STAGE"
