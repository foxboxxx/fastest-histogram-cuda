import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# CUDA source code loaded from submission.cu
cuda_source = """
#include <cuda_runtime.h>

#define MAX_SHARED_MEMORY (47 * 1024)
#define MAX_SHARED_MEMORY_OPT_IN (227 * 1024)

/*
 * threadId --> thread #
 * blockDim --> # of threads per block
 * blockIdx --> block #
 */
__global__ void HistogramKernel(const uint8_t* data_in, int32_t* data_out, int length, int num_channels, int num_bins, int channels_per_block) {
    extern __shared__ int32_t shared_mem[]; 
    // zero out the shared histogram
    for (int i = threadIdx.x; i < channels_per_block * num_bins; i += blockDim.x) {
        shared_mem[i] = 0;
    }
    __syncthreads();

    // determine global index of thread (# from 0 --> # of values / channels_per_block)
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int starting_channel = (global_idx / length) * channels_per_block;
    int row = global_idx % length;

    // perform channels_per_block iterations in this thread to promote spatial locality and l1 cache hits
    for (int local_channel = 0; local_channel < channels_per_block; local_channel++) {
        int channel = starting_channel + local_channel;
        uint8_t value = data_in[row * num_channels + channel];
        atomicAdd(&shared_mem[local_channel * num_bins + value], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < channels_per_block * num_bins; i += blockDim.x) {
        int channel = starting_channel + (i / num_bins);
        int bin = (i % num_bins);
        int32_t count = shared_mem[i];
        if (count > 0) {
            atomicAdd(&data_out[channel * num_bins + bin], count);
        }
    }

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


    const int channels_per_block = 128;

    // H100 have 1024 threads per block 
    const int threads = 1024; 

    // 144 // 132 // 114 SMs. 
    const int blocks = (length * num_channels + (threads * channels_per_block) - 1) / (threads * channels_per_block);

    // 
    size_t shared_mem_size = channels_per_block * (num_bins) * sizeof(int32_t);
    cudaFuncSetAttribute(HistogramKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    // Officially launch da kernel
    HistogramKernel<<<blocks, threads, shared_mem_size>>>(data_in, data_out, length, num_channels, num_bins, channels_per_block);
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
