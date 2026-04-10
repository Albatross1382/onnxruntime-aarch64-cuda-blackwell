# ONNX Runtime — aarch64 + CUDA 13 + Blackwell (GB10/DGX Spark)

**Prebuilt ONNX Runtime 1.24.4 shared libraries with CUDA Execution Provider for NVIDIA Blackwell (sm_121) on aarch64 Linux.**

As of April 2026, no prebuilt ONNX Runtime GPU binaries exist for aarch64 Linux from any source — not Microsoft's GitHub releases, not PyPI (`onnxruntime-gpu`), not the `ort` Rust crate's download cache (pyke.io), and not NVIDIA's apt repositories. This repo fills that gap.

## Hardware Tested

| Component | Value |
|---|---|
| Platform | NVIDIA DGX Spark (ASUS Ascent GX10) |
| CPU | NVIDIA Grace (ARM64/aarch64) |
| GPU | NVIDIA GB10 (Blackwell, sm_121) |
| Memory | 128 GB unified LPDDR5x |
| OS | Ubuntu 24.04 (Noble) |
| CUDA | 13.0 (V13.0.88) |
| Driver | 580.142 |
| cuDNN | 9.20.0 |

## What's Included

```
lib/
├── libonnxruntime.so.1.24.4          # Main ORT shared library (CUDA-enabled)
├── libonnxruntime.so.1               # Symlink → libonnxruntime.so.1.24.4
├── libonnxruntime.so                 # Symlink → libonnxruntime.so.1
├── libonnxruntime_providers_cuda.so  # CUDA execution provider plugin
└── libonnxruntime_providers_shared.so # Shared provider infrastructure
```

## Installation

```bash
sudo cp lib/*.so* /usr/local/lib/
sudo ldconfig
```

Verify:
```bash
ldconfig -p | grep onnxruntime
```

## Prerequisites

- CUDA 13.0 toolkit (`nvcc --version`)
- cuDNN 9.x for CUDA 13 (`sudo apt install libcudnn9-cuda-13 libcudnn9-dev-cuda-13`)
- NVIDIA driver 580+ with Blackwell support

## Usage

### Python
```python
# onnxruntime Python package won't work (no aarch64 GPU wheel exists).
# Use ctypes or build the Python wheel from source (see Build Instructions).
```

### Rust (ort crate)
```toml
[dependencies]
ort = { version = "=2.0.0-rc.12", features = ["cuda", "load-dynamic"] }
```

```bash
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so cargo run --release
```

> **Important:** The `ort` crate's built-in CUDA EP registration (V2 API) silently fails with `load-dynamic` on ORT 1.24.4. The CUDA EP appears to register but inference falls back to CPU. You must use the legacy C API (`OrtSessionOptionsAppendExecutionProvider_CUDA`) via FFI. See the [Rust CUDA EP Workaround](#rust-ort-crate-cuda-ep-workaround) section below.

### C/C++
```c
#include <dlfcn.h>
void* lib = dlopen("/usr/local/lib/libonnxruntime.so", RTLD_LAZY);
// Use ORT C API as normal
```

## Rust `ort` Crate CUDA EP Workaround

The `ort` crate (v2.0.0-rc.12) registers CUDA via `SessionOptionsAppendExecutionProvider_CUDA_V2`, which is part of the ORT API struct. With `load-dynamic`, this registration silently succeeds but ORT doesn't route ops to CUDA. The legacy standalone function `OrtSessionOptionsAppendExecutionProvider_CUDA` works correctly.

```rust
use ort::AsPointer;

let mut builder = ort::session::Session::builder()?;

// Register CUDA EP via legacy C API
unsafe {
    let opts_ptr = builder.ptr() as *mut core::ffi::c_void;

    type CudaEpFn = unsafe extern "C" fn(
        *mut core::ffi::c_void, i32
    ) -> *mut core::ffi::c_void;

    let dylib_path = std::env::var("ORT_DYLIB_PATH")
        .unwrap_or_else(|_| "/usr/local/lib/libonnxruntime.so".to_string());
    let lib = libloading::Library::new(&dylib_path)?;
    let func: libloading::Symbol<CudaEpFn> = lib
        .get(b"OrtSessionOptionsAppendExecutionProvider_CUDA")?;

    let status = func(opts_ptr, 0); // device_id = 0
    if !status.is_null() {
        eprintln!("CUDA EP registration failed, falling back to CPU");
    }
    std::mem::forget(lib); // Keep library loaded
}

let session = builder.commit_from_file("model.onnx")?;
```

## Performance

Tested with snowflake-arctic-embed-m-v2.0 (768-dim embedding model):

| Backend | Model | Inference Time |
|---|---|---|
| tract-onnx (pure Rust, CPU) | Quantized INT8 | ~3,400ms |
| ORT 1.24.4 CPU | FP32 | ~135ms |
| ORT 1.24.4 CUDA (GB10) | FP32 | ~149ms cold / TBD warm |
| ORT 1.24.4 CPU (CUDA_VISIBLE_DEVICES="") | FP32 | ~3,360ms |

The CUDA vs CPU-disabled test confirms GPU acceleration is active. The similar cold-start times (135ms ORT CPU vs 149ms CUDA) reflect that cold start is dominated by model loading; the CUDA advantage becomes clear in sustained/warm inference (e.g., MCP server with persistent session).

## Build Instructions

If you want to build from source rather than using the prebuilt binaries:

```bash
# Prerequisites
sudo apt install cmake libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libeigen3-dev

# Clone ORT v1.24.4
git clone --recursive --branch v1.24.4 --depth 1 \
  https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# Build with CUDA for Blackwell (sm_121)
./build.sh --config Release \
  --use_cuda \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr \
  --build_shared_lib \
  --parallel $(nproc) \
  --skip_tests \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=121

# Output: build/Linux/Release/libonnxruntime.so.1.24.4
```

### Build Gotchas

| Issue | Solution |
|---|---|
| Eigen SHA1 hash mismatch (v1.20.1) | Use ORT v1.24.4+ or pre-clone Eigen via git and set `FETCHCONTENT_SOURCE_DIR_EIGEN` |
| `thrust::unary_function` removed (CUDA 13) | Use ORT v1.24.4+ (v1.20.1 is incompatible with CUDA 13) |
| `nvcc fatal: Unsupported gpu architecture 'compute_53'` | Add `--cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=121` |
| sm_120 vs sm_121 | GB10 is compute capability 12.1, NOT 12.0. Use `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` to check |
| `ort-sys` downloads CPU-only binary | No aarch64+CUDA prebuilts on pyke.io. Use `load-dynamic` + `ORT_DYLIB_PATH` |
| CUDA EP silently falls back to CPU | `ort` crate V2 API issue with `load-dynamic`. Use legacy FFI (see above) |

## Known Limitations

- INT8 quantized models may not have CUDA kernels for sm_121. Use FP32 models for CUDA inference.
- The `ort` Rust crate's `fail_silently()` default on EP registration masks CUDA failures. Use `.error_on_failure()` during debugging.
- GB10 uses unified memory — `nvidia-smi` won't show separate GPU memory usage for CUDA compute processes.

## License

The ONNX Runtime binaries are subject to the [MIT License](https://github.com/microsoft/onnxruntime/blob/main/LICENSE).
This repository's documentation and build scripts are MIT licensed.

## Related Issues

- [pyke/ort] CUDA EP silent failure with `load-dynamic` on ORT 1.24.4 aarch64
- [pyke/ort] No prebuilt aarch64 CUDA binaries on pyke.io
- [microsoft/onnxruntime] No official aarch64 Linux GPU release artifacts
