#!/bin/bash
set -eux

echo "=== Rebuilding DeepEP with kNumMaxTopK=16 for Qwen3.5 (topk=10) ==="

DEEPEP_SRC="/sgl-workspace/DeepEP"

if [ ! -d "$DEEPEP_SRC" ]; then
    echo "ERROR: DeepEP source not found at $DEEPEP_SRC (mount via extra_mount)"
    exit 1
fi

cd "$DEEPEP_SRC"

# Find NVSHMEM
NVSHMEM_DIR=$(find /usr/local -name "nvshmem" -type d 2>/dev/null | head -1)
echo "NVSHMEM_DIR=$NVSHMEM_DIR"

# Fix missing nvshmem symlinks (container has .so.3 but not .so)
NVSHMEM_LIB="$NVSHMEM_DIR/lib"
if [ ! -f "$NVSHMEM_LIB/libnvshmem_host.so" ] && [ -f "$NVSHMEM_LIB/libnvshmem_host.so.3" ]; then
    echo "Creating missing nvshmem symlinks..."
    ln -sf libnvshmem_host.so.3 "$NVSHMEM_LIB/libnvshmem_host.so"
fi

# Verify our patch is in place
grep -q "kNumMaxTop. = 16" csrc/kernels/internode_ll.cu && echo "Patch verified: kNumMaxTopK/k=16" || {
    echo "ERROR: kNumMaxTopK patch not found in source!"; exit 1;
}

# Build with full output so we can debug failures
TORCH_CUDA_ARCH_LIST="10.0" \
NVSHMEM_DIR="$NVSHMEM_DIR" \
pip install -e . --no-build-isolation 2>&1

if [ $? -ne 0 ]; then
    echo "=== DeepEP rebuild FAILED ==="
    exit 1
fi

echo "=== DeepEP rebuild complete ==="
python3 -c "import deep_ep; print('deep_ep imported successfully')"
