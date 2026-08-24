# ferrox-cuda

CUDA/NVRTC kernels for Ferrox.

Capability detection always builds. Real CUDA dispatch sits behind
`--features cuda`. Loading is dynamic, through `cudarc`, so no toolkit
is needed to compile. Run the hardware tests only on a machine with a
GPU.
