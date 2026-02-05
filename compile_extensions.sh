#!/bin/bash
# Compile C++/CUDA extensions for Hunyuan paint pipeline
# Run this after installing dependencies (torch, pybind11, etc.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Compiling Hunyuan paint pipeline extensions..."

# 1. Compile mesh_inpaint_processor (pybind11 C++ extension)
echo ""
echo "1. Compiling mesh_inpaint_processor..."
cd modules/hunyuan/paint/DifferentiableRenderer
if bash compile_mesh_painter.sh; then
    echo "   ✓ mesh_inpaint_processor compiled successfully"
    ls -lh mesh_inpaint_processor*.so 2>/dev/null || echo "   Warning: .so file not found"
else
    echo "   ✗ mesh_inpaint_processor compilation failed (has fallback)"
fi

# 2. Compile custom_rasterizer (CUDA extension) - REQUIRED
echo ""
echo "2. Compiling custom_rasterizer (CUDA extension)..."
cd "$SCRIPT_DIR/modules/hunyuan/paint/custom_rasterizer"
if pip install --no-build-isolation -e .; then
    echo "   ✓ custom_rasterizer compiled successfully"
else
    echo "   ✗ custom_rasterizer compilation failed - this is REQUIRED for paint pipeline!"
    exit 1
fi

echo ""
echo "✓ All extensions compiled successfully!"
