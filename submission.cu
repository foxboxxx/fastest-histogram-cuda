#include <cuda_runtime.h>

#define MAX_SHARED_MEMORY (47 * 1024)
#define MAX_SHARED_MEMORY_OPT_IN (227 * 1024)

/*
 * threadIdx.x --> thread #
 * blockDim.x --> # of threads per block
 * blockIdx.x --> block #
 * gridDim.x --> # of blocks
 */
__global__ void HistogramKernel(const uint8_t* __restrict__ data_in, int32_t* __restrict__ data_out, int length, int num_channels, int num_bins, int channels_per_batch, int num_batches) {
    extern __shared__ int32_t shared_mem[]; 
    int total_threads = blockDim.x * gridDim.x;
    int batch = blockIdx.y; 
    if (batch >= num_batches) return; 

    // for (int batch = 0; batch < num_batches; batch++) {
    int channel_offset = batch * channels_per_batch;

    // clear memory
    for (int i = threadIdx.x; i < channels_per_batch * num_bins; i += blockDim.x) {
        shared_mem[i] = 0;
    }
    __syncthreads();

    // calculate thread index, max vectors per row, and total vectors to compute
    int current_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    // int vectors_per_row = channels_per_batch / 16; 
    int vectors_per_row = channels_per_batch >> 4;
    int total_vectors = length * vectors_per_row;

    // loop through until all vectors are processed 
    for (int i = current_thread_idx; i < total_vectors; i += total_threads) {
        // int row = i / vectors_per_row;
        // int vec_idx = i % vectors_per_row;
        int row = i >> 2;
        int vec_idx = i & (vectors_per_row - 1);
        int base_local_channel = vec_idx * 16;
        
        int addr_offset = row * num_channels + (base_local_channel);

        uchar4* vec_addr = (uchar4*)(data_in + channel_offset + addr_offset); 
        uint4 vec_val = *reinterpret_cast<const uint4*>(vec_addr);
        uint8_t* vec_converted = reinterpret_cast<uint8_t*>(&vec_val);
        
        #pragma unroll
        for (int elem = 0; elem < 16; elem++) {
            uint8_t val = vec_converted[elem];
            int local_channel = base_local_channel + elem;
            atomicAdd(&shared_mem[local_channel * num_bins + val], 1);
        }
    }
    __syncthreads();

    // atomic add back to global histogram
    for (int i = threadIdx.x; i < channels_per_batch * num_bins; i += blockDim.x) {
        int32_t count = shared_mem[i];
        if (count > 0) {
            // int channel = channel_offset + (i / num_bins);
            // int bin = (i % num_bins);
            int channel = channel_offset + (i >> 8); // log256
            int bin = (i & 255); // same as mod 256
            atomicAdd(&data_out[channel * num_bins + bin], count);
        }
    }

    // }
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


    const int channels_per_batch = 64;

    // H100 have 1024 threads per block 
    const int threads = 1024; 

    // 144 // 132 // 114 SMs. 
    // const int blocks = 132 * 2;

    // 
    size_t shared_mem_size = channels_per_batch * (num_bins) * sizeof(int32_t);
    cudaFuncSetAttribute(HistogramKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    const int num_batches = (num_channels + channels_per_batch - 1) / channels_per_batch;
    dim3 blocks(132, num_batches);

    // Officially launch da kernel
    HistogramKernel<<<blocks, threads, shared_mem_size>>>(data_in, data_out, length, num_channels, num_bins, channels_per_batch, num_batches);
    ////

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return histogram;
}