#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building tinybvh WASM..."
mkdir -p build
cd build

emcmake cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

echo "Copying artifacts to example/libs/..."
mkdir -p ../../example/libs
cp tinybvh.js tinybvh.wasm ../../example/libs/

echo "Done! Built tinybvh.js + tinybvh.wasm"
ls -lh ../../example/libs/tinybvh.*
