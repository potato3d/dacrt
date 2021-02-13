# dacrt
Divide-and-conquer ray tracing

These are some of my own implementations of the article [Naive ray-tracing: A divide-and-conquer approach](https://dl.acm.org/doi/10.1145/2019627.2019636)

* dacrt: basic CPU version
* dacrt_simd: more optimized SIMD version
* dacrt_async: experiments doing the spatial subdivision on the CPU and the ray-triangle intersection on the GPU (using CUDA)

The main bottleneck of this last approach was the logical dependencies between the CPU "traversal" and GPU intersections. We need to wait for the intersection results on the GPU in order to decide if we need to proceed with spatial traversal on the CPU and vice-versa.