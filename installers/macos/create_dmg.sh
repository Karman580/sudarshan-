#!/bin/bash
# ==============================================================================
# SUDARSHAN AI - macOS DMG Packaging & Styling Script
# ==============================================================================
# Usage: ./create_dmg.sh <path_to_app> <output_dmg_path>
# ==============================================================================

set -e

APP_PATH="$1"
OUT_DMG="$2"
VOL_NAME="SUDARSHAN AI Installer"
BG_PNG="../../assets/dmg_background.png"

if [ -z "$APP_PATH" ] || [ -z "$OUT_DMG" ]; then
    echo "Usage: $0 <path_to_app> <output_dmg_path>"
    exit 1
fi

# Ensure paths are absolute
APP_PATH=$(cd "$(dirname "$APP_PATH")" && pwd)/$(basename "$APP_PATH")
OUT_DIR=$(cd "$(dirname "$OUT_DMG")" && pwd)
OUT_DMG="$OUT_DIR/$(basename "$OUT_DMG")"
BG_PNG=$(cd "$(dirname "$BG_PNG")" && pwd)/$(basename "$BG_PNG")

TMP_DIR="dmg_temp"
rm -rf "$TMP_DIR" temp.dmg
mkdir -p "$TMP_DIR"

echo "[DMG] Copying Application Bundle..."
cp -R "$APP_PATH" "$TMP_DIR/"

echo "[DMG] Creating Applications Symlink..."
ln -s /Applications "$TMP_DIR/Applications"

echo "[DMG] Creating writeable temporary disk image..."
hdiutil create -srcfolder "$TMP_DIR" -volname "$VOL_NAME" -fs HFS+ -format UDRW temp.dmg > /dev/null

echo "[DMG] Mounting temporary disk image..."
# Mount the raw disk image and capture the device name
device=$(hdiutil attach -readwrite -noverify temp.dmg | egrep '^/dev/' | sed 1q | awk '{print $1}')
sleep 2

VOL_PATH="/Volumes/$VOL_NAME"

echo "[DMG] Copying hidden background image..."
mkdir -p "$VOL_PATH/.background"
cp "$BG_PNG" "$VOL_PATH/.background/dmg_background.png"

echo "[DMG] Applying AppleScript Finder window layout styling..."
# Inject AppleScript to style the Finder view
osascript <<APPLESCRIPT
tell application "Finder"
    tell disk "$VOL_NAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        
        # Position window: {left, top, right, bottom}
        set the bounds of container window to {400, 100, 1000, 500}
        
        set theViewOptions to the icon view options of container window
        set icon size of theViewOptions to 80
        set arrangement of theViewOptions to not arranged
        
        # Set custom background image
        set background picture of theViewOptions to file ".background:dmg_background.png"
        
        # Position the interactive icons {X, Y}
        set position of item "SUDARSHAN AI.app" of container window to {150, 180}
        set position of item "Applications" of container window to {450, 180}
        
        update without registering applications
        delay 2
        close
    end tell
end tell
APPLESCRIPT

# Force update and flush changes
sync

echo "[DMG] Detaching temporary disk volume..."
hdiutil detach "$device"
sleep 1

echo "[DMG] Compressing into final read-only production DMG..."
if [ -f "$OUT_DMG" ]; then
    rm -f "$OUT_DMG"
fi
hdiutil convert temp.dmg -format UDZO -imagekey zlib-level=9 -o "$OUT_DMG" > /dev/null

# Clean up temp resources
echo "[DMG] Cleaning up build temporary directories..."
rm -f temp.dmg
rm -rf "$TMP_DIR"

echo "[DMG SUCCESS] Generated branded DMG installer: $(basename "$OUT_DMG")"
