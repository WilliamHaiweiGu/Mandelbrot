#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>


using namespace std;

constexpr unsigned int MAX_ITER = 10000;
constexpr unsigned int W_PIX = 65536;
constexpr unsigned int H_PIX = 65536;
constexpr unsigned int X_PIX_PER_BLK = 32;
const string DATA_FILE_NAME = "out.dat";

constexpr unsigned int H_PIX_HALF = H_PIX / 2;

__device__ inline bool run_iter(const double x0, const double y0)
{
    double x = x0;
    double y = y0;
    for (int i = 0; i < MAX_ITER; i++)
    {
        const double x2 = x * x;
        const double y2 = y * y;
        const double hyp2 = x2 + y2;
        if (i == 0 && hyp2 < 1.0 / 16)
            return true;
        if (hyp2 > 4)
            return false;
        y = 2 * x * y + y0;
        x = x2 - y2 + x0;
    }
    return true;
}

__global__ void iter_kernel(unsigned char *device_mem, double x_min, double w, double y_min, double h)
{
    const int pix_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pix_y > H_PIX_HALF)
        return;
    const int pix_x = threadIdx.x + blockIdx.x * blockDim.x;
    const double x = pix_x * w / W_PIX + x_min;
    const double y = pix_y * h / H_PIX + y_min;
    const int pix_value = run_iter(x, y) ? 0 : 255;
    device_mem[pix_y * W_PIX + pix_x] = static_cast<unsigned char>(pix_value);
}

int main()
{
    int device_cnt;
    cudaGetDeviceCount(&device_cnt);
    if (device_cnt <= 0)
    {
        std::cout << "No GPU detected " << endl;
        return 0;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int max_parallel_thread = prop.multiProcessorCount * min(prop.maxThreadsPerBlock * prop.maxBlocksPerMultiProcessor, prop.maxThreadsPerMultiProcessor);
    cout << "GPU: \"" << prop.name << "\"\n"
         << " Max threads per block: " << prop.maxThreadsPerBlock << '\n'
         << " Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << '\n'
         << " Max threads per SM: " << prop.maxThreadsPerMultiProcessor << '\n'
         << " Number of SMs: " << prop.multiProcessorCount << '\n'
         << " Max Parallel Threads on Device: " << max_parallel_thread << '\n';
    

    ofstream file("out.dat", ios::out | ios::binary);
    if (!file) {
        cerr << "Cannot open " << DATA_FILE_NAME << endl;
        return 1;
    }
    

    const int y_pix_per_blk = prop.maxThreadsPerBlock / X_PIX_PER_BLK;
    const dim3 blk_dim(X_PIX_PER_BLK, y_pix_per_blk);
    const int y_grid_dim = (H_PIX / y_pix_per_blk) / 2 + 1;
    const dim3 grid_dim(W_PIX / X_PIX_PER_BLK, y_grid_dim);
    unsigned char *mem_device;
    const unsigned int n_pix = W_PIX * (H_PIX_HALF + 1);
    const size_t size_of_mem = n_pix * sizeof(unsigned char);
    cudaMalloc((void **)&mem_device, size_of_mem);
    
    auto start = chrono::high_resolution_clock::now();
    iter_kernel<<<grid_dim, blk_dim>>>(mem_device, -2, 4, -2, 4);
    cudaDeviceSynchronize();
    auto stop = chrono::high_resolution_clock::now();
    cout << "GPU Time: " << chrono::duration_cast<chrono::microseconds>(stop - start).count() / 1e6 << " seconds" << endl;
    
    unsigned char *mem_host = new unsigned char[n_pix];
    cudaMemcpy(mem_host, mem_device, size_of_mem, cudaMemcpyDeviceToHost);
    cudaFree(mem_device);

    file.write(reinterpret_cast<const char*>(mem_host), size_of_mem);
    delete[] mem_host;
    file.close();
    cout << "Data written to " << DATA_FILE_NAME << endl;
    return 0;
}
