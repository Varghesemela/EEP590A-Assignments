
#include <hip/hip_runtime.h>
#include <wb.h>

#define TILE_SIZE 4

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float tileM[TILE_SIZE][TILE_SIZE];
  __shared__ float tileN[TILE_SIZE][TILE_SIZE];
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;

  float result = 0;
  for(int i = 0; i < ceil(1.0*numAColumns/TILE_SIZE); i++) {
    int a_index = i * TILE_SIZE + tid_x;
    int b_index = i * TILE_SIZE + tid_y;
    
    if (row < numARows && (a_index)<numAColumns){
      tileM[tid_y][tid_x] = A[row * numAColumns + a_index];
    } else {
      tileM[tid_y][tid_x] = 0;
    }
    
    if ((b_index < numBRows) && col < numBColumns){
      tileN[tid_y][tid_x] = B[b_index * numBColumns + col];
    } else{
      tileN[tid_y][tid_x] = 0;
    }
    __syncthreads();
    for (int j = 0; j < TILE_SIZE; j++) {
      result += tileM[tid_y][j] * tileN[j][tid_x];
      __syncthreads();
    }

  }
  if (row < numCRows && col < numCColumns){
    C[row * numCColumns + col] = result;
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
  hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  long long numASize = numARows * numAColumns;
  long long numBSize = numBRows * numBColumns;
  long long numCSize = numCRows * numCColumns;

  hipMalloc((void**) &deviceA, numASize * sizeof(float));
  hipMalloc((void**) &deviceB, numBSize * sizeof(float));
  hipMalloc((void**) &deviceC, numCSize * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  hipMemcpy(deviceA, hostA, numASize * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(deviceB, hostB, numBSize * sizeof(float), hipMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((numCColumns/(double)TILE_SIZE)), ceil((numCRows/(double)TILE_SIZE)), 1);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

  wbTime_start(Compute, "Performing HIP computation");
  //@@ Launch the GPU Kernel here
  hipLaunchKernelGGL(matrixMultiplyShared, dimGrid, dimBlock, 0, 0, 
                      deviceA, deviceB, deviceC,
                      numARows, numAColumns,
                      numBRows, numBColumns,
                      numCRows, numCColumns
                    );
  hipDeviceSynchronize();
  wbTime_stop(Compute, "Performing HIP computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  hipMemcpy(hostC, deviceC, numCSize * sizeof(float), hipMemcpyDeviceToHost);

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
