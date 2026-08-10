# ferrox-cuda

CUDA/NVRTC kernels for Ferrox.

Capability detection always builds; real CUDA dispatch is behind
`--features cuda`. Uses `cudarc` dynamic loading (no toolkit required
to compile). Run hardware tests only on a machine with a GPU.
