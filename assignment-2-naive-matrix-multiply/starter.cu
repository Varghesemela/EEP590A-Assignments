#include <hip/hip_runtime.h>
#include <wb.h>
#include <stdint.h>

#define DIM_TILE 16
#define DIM_BLOCK(t) (int)(DIM_TILE + t - 1)/DIM_TILE
  

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void matrixTranspose(int* in, int* out, unsigned int rows, unsigned int cols) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows){
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        out[trans_pos] = in[pos];
    }
}

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, 
                                int numARows, int numAColumns, 
                                int numBRows, int numBColumns, 
                                int numCRows, int numCColumns
                                ) {
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t sharedDim = numAColumns;
  float tempVal = 0.0f;
  if(row < numCRows && col < numCColumns){
    size_t offset = row * numCColumns + col;
    for(size_t n = 0; n < sharedDim; n++){
      tempVal += A[row * sharedDim + n] * B[col + n * numCColumns]; 
    }
    C[offset] = tempVal;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  if (hostC == NULL){
    wbLog(ERROR, "Couldnt allocate memory for host output buffer\n");      \
    return -1;
  } 

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  size_t totalElementsSizeA = numARows * numAColumns * sizeof(float);
  size_t totalElementsSizeB = numBRows * numBColumns * sizeof(float);
  size_t totalElementsSizeC = numCRows * numCColumns * sizeof(float);
  hipMalloc((void **)&deviceA, totalElementsSizeA);
  hipMalloc((void **)&deviceB, totalElementsSizeB);
  hipMalloc((void **)&deviceC, totalElementsSizeC);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  hipStream_t streamA;
  hipStream_t streamB;
  hipStreamCreate(&streamA);
  hipStreamCreate(&streamB);
  
  hipMemcpyAsync(deviceA, hostA, totalElementsSizeA, hipMemcpyHostToDevice, streamA);
  hipMemcpyAsync(deviceB, hostB, totalElementsSizeB, hipMemcpyHostToDevice, streamB);
  hipDeviceSynchronize();
  hipStreamDestroy(streamA);
  hipStreamDestroy(streamB);
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 numthreads(DIM_TILE, DIM_TILE, 1);
  dim3 numblocks((uint32_t)DIM_BLOCK(numCColumns), (uint32_t)DIM_BLOCK(numCRows), 1);

  wbTime_start(Compute, "Performing HIP computation");
  //@@ Launch the GPU Kernel here
  hipLaunchKernelGGL(matrixMultiply, numblocks, numthreads, 0, 0, 
                                      deviceA, deviceB, deviceC,
                                      numARows, numAColumns,
                                      numBRows, numBColumns,
                                      numCRows, numCColumns 
  );

  hipDeviceSynchronize();
  wbTime_stop(Compute, "Performing HIP computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  hipMemcpy(hostC, deviceC, totalElementsSizeC, hipMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  hipFree(deviceA);
  hipFree(deviceB);
  hipFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
