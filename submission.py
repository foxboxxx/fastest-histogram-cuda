import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# CUDA source code loaded from submission.cu
cuda_source = """
#include <cuda_runtime.h>

#define MAX_SHARED_MEMORY (47 * 1024)

//
// create your function: __global__ void kernel(...) here
// Note: input data is of type uint8_t
//
__global__ void HistogramKernel(const uint8_t* __restrict__ data_in, int32_t* __restrict__ data_out, int length, int num_channels, int num_bins) {
    // Determine the current block's channel
    int channel = blockIdx.x;
    if (channel >= num_channels) return;

    // Establish shraed memory and set to 0 (then sync all threads so nothing is modifying while everything is set)
    extern __shared__ int32_t shared_histogram[]; 
    #pragma unroll
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
        shared_histogram[i] = 0;
    __syncthreads();

    // Perform atomic add on the local shared memory within the block
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        uint8_t value = data_in[i * num_channels + channel];
        if (value < num_bins) atomicAdd(&shared_histogram[value], 1);
    }

    // Sync threads in the block and migrate data to output array
    __syncthreads();
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) 
        atomicAdd(&data_out[channel * num_bins + i], shared_histogram[i]);
}

// Host function to launch kernel
torch::Tensor
histogram_kernel(torch::Tensor data, // [length, num_channels], dtype=uint8
                 int num_bins) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");

    const int length = data.size(0);
    const int num_channels = data.size(1);

    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);

    ////
    // Launch your kernel here
    const uint8_t* data_in = data.data_ptr<uint8_t>();
    int32_t* data_out = histogram.data_ptr<int32_t>();

    // H100 have 1024 threads per block 
    const int threads = 256; // fastest when threads = 256 for channel-tiling implementation
    // const int blocks = (length * num_channels + threads - 1) / threads;
    const int blocks = num_channels;

    // Establish shared memory size per block to perform faster atomic adds within each block
    size_t shared_mem_size = num_bins * sizeof(int32_t);

    // Officially launch da kernel
    HistogramKernel<<<blocks, threads, shared_mem_size>>>(data_in, data_out, length, num_channels, num_bins);
    ////

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return histogram;
}
"""

# C++ header declaration
cpp_source = """
#include <torch/extension.h>
torch::Tensor histogram_kernel(torch::Tensor data, int num_bins);
"""

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

cuda_module = load_inline(
    name='submission_cuda_histogram_tazzi',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['histogram_kernel'],
    verbose=True,  # Enable verbose to see compilation details
    # with_cuda=True,
    # build_directory=".",
)

def custom_kernel(data: input_t) -> output_t:
    """
    Wrapper function matching the required signature.
    
    Args:
        data: Tuple of (array, num_bins) where:
            array:    Tensor of shape [length, num_channels] with integer values in [0, num_bins-1]
            num_bins: Number of bins for the histogram
    
    Returns:
        histogram: Tensor of shape [num_channels, num_bins] containing histogram counts for each channel
    """

    array, num_bins = data
    
    if not array.is_cuda:
        array = array.cuda()
    
    # Call CUDA kernel
    histogram = cuda_module.histogram_kernel(array, num_bins)

    return histogram
