#!/bin/bash
# Ensure working directory is the script directory
cd "$(dirname "$0")"

SRC_PNG="../../assets/app_icon_source.png"
ICONSET="AppIcon.iconset"

if [ ! -f "$SRC_PNG" ]; then
    echo "Error: Source image $SRC_PNG not found!"
    exit 1
fi

echo "Creating iconset folder..."
mkdir -p "$ICONSET"

echo "Scaling images using sips..."
sips -s format png -z 16 16     "$SRC_PNG" --out "$ICONSET/icon_16x16.png" > /dev/null 2>&1
sips -s format png -z 32 32     "$SRC_PNG" --out "$ICONSET/icon_16x16@2x.png" > /dev/null 2>&1
sips -s format png -z 32 32     "$SRC_PNG" --out "$ICONSET/icon_32x32.png" > /dev/null 2>&1
sips -s format png -z 64 64     "$SRC_PNG" --out "$ICONSET/icon_32x32@2x.png" > /dev/null 2>&1
sips -s format png -z 128 128   "$SRC_PNG" --out "$ICONSET/icon_128x128.png" > /dev/null 2>&1
sips -s format png -z 256 256   "$SRC_PNG" --out "$ICONSET/icon_128x128@2x.png" > /dev/null 2>&1
sips -s format png -z 256 256   "$SRC_PNG" --out "$ICONSET/icon_256x256.png" > /dev/null 2>&1
sips -s format png -z 512 512   "$SRC_PNG" --out "$ICONSET/icon_256x256@2x.png" > /dev/null 2>&1
sips -s format png -z 512 512   "$SRC_PNG" --out "$ICONSET/icon_512x512.png" > /dev/null 2>&1
sips -s format png -z 1024 1024 "$SRC_PNG" --out "$ICONSET/icon_512x512@2x.png" > /dev/null 2>&1

echo "Compiling AppIcon.icns using iconutil..."
iconutil -c icns "$ICONSET"

echo "Cleaning up temporary iconset directory..."
rm -rf "$ICONSET"

echo "Successfully generated AppIcon.icns!"
