#!/bin/zsh
# Compile the Icon-Composer-style AppIcon.icon bundle (light + dark layers)
# and inject it into a built Donkey simulator .app, so macOS swaps the
# dock/Finder icon with the system appearance (light = the shipped white-
# background artwork; dark = the same artwork on a pure #000000 background).
#
# Technique and hard-won findings ported from the PathBlender pipeline
# (2026-10) and re-verified there on macOS 27:
# - A plain Unity PlayerIcon.icns (CFBundleIconFile) carries no appearance
#   variants; classic .appiconset luminosity entries are silently flattened
#   by actool, so they can't carry a dark icon either.
# - The .icon bundle does: actool compiles it into an Assets.car whose
#   AppIcon iconstack carries Aqua + DarkAqua renditions, and macOS prefers
#   the icon named by CFBundleIconName over the legacy icns.
# - The layer MUST carry "glass": false (plus blend-mode/fill, group
#   translucency and supported-platforms, i.e. the canonical set Icon
#   Composer itself writes); the group-level "shadow" key must be omitted
#   entirely (a kind:"none" value makes actool silently produce no
#   Assets.car). Do not strip or add keys.
#
# The legacy PlayerIcon.icns stays in place as the fallback (e.g. for hosts
# that ignore the asset catalog). Unity player apps are not sealed by
# default; if the target IS signed, re-sign after this script.
#
# Usage: tools/simulator-icon/apply-dark-icon.sh <path-to-donkey_sim.app>
set -euo pipefail
cd "$(dirname "$0")/../.."

APP="${1:?usage: apply-dark-icon.sh <path-to-donkey_sim.app>}"
[[ -d "$APP/Contents/MacOS" ]] || { echo "apply-dark-icon.sh: not a macOS app bundle: $APP" >&2; exit 1; }

ICON_BUNDLE="tools/simulator-icon/AppIcon.icon"
[[ -f "$ICON_BUNDLE/icon.json" ]] || { echo "apply-dark-icon.sh: icon bundle missing: $ICON_BUNDLE" >&2; exit 1; }

# actool ships with full Xcode; the Command Line Tools alone don't have it.
if ! ACTOOL="$(xcrun --find actool 2>/dev/null)"; then
  for X in /Applications/Xcode.app /Applications/Xcode-beta.app; do
    if [[ -x "$X/Contents/Developer/usr/bin/actool" ]]; then
      export DEVELOPER_DIR="$X/Contents/Developer"
      ACTOOL="$(xcrun --find actool)"
      break
    fi
  done
fi
[[ -n "${ACTOOL:-}" ]] || { echo "apply-dark-icon.sh: actool not found (needs full Xcode, not just CLT)" >&2; exit 1; }

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

"$ACTOOL" "$ICON_BUNDLE" --app-icon AppIcon \
  --compile "$OUT" \
  --output-partial-info-plist "$OUT/partial.plist" \
  --minimum-deployment-target 13.0 --platform macosx --target-device mac \
  >/dev/null

CAR="$OUT/Assets.car"
[[ -s "$CAR" ]] || { echo "apply-dark-icon.sh: actool produced no Assets.car" >&2; exit 1; }

# Hard gate: the compiled catalog must actually carry a dark-appearance
# iconstack rendition, otherwise the injection would silently install a
# light-only icon.
if ASSETUTIL="$(xcrun --find assetutil 2>/dev/null)"; then
  DARK_COUNT=$("$ASSETUTIL" --info "$CAR" | grep -c 'NSAppearanceNameDarkAqua')
  [[ "$DARK_COUNT" -ge 1 ]] || { echo "apply-dark-icon.sh: Assets.car carries no dark appearance rendition" >&2; exit 1; }
fi

cp "$CAR" "$APP/Contents/Resources/Assets.car"
/usr/libexec/PlistBuddy -c 'Delete :CFBundleIconName' "$APP/Contents/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c 'Add :CFBundleIconName string AppIcon' "$APP/Contents/Info.plist"

# macOS only honors the Assets.car appearance catalog for SIGNED bundles —
# an unsigned Unity player silently falls back to the legacy icns in both
# appearances (verified live 2026-11-18: ad-hoc signing immediately turned
# the dual rendition on). Re-seal ad-hoc after injection; a distribution
# identity, if used, should re-sign afterwards as usual.
if ! codesign --verify "$APP" 2>/dev/null; then
  codesign --force --deep --sign - "$APP"
  codesign --verify --deep --strict "$APP"
  echo "apply-dark-icon.sh: re-sealed ad-hoc (was unsigned)"
fi

echo "apply-dark-icon.sh: light+dark AppIcon catalog injected into $APP"
