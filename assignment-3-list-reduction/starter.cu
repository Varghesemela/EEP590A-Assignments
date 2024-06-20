// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <hip/hip_runtime.h>
#include <wb.h>

//BLOCK_SIZE is NUM THREADS
#define BLOCK_SIZE 512 //@@ You can change this
#define MASK 0xffffffff

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void total_basic(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  __shared__ float sharedMem[2*BLOCK_SIZE];
  unsigned int tid = threadIdx.x;
  unsigned int offset = 2 * blockIdx.x * blockDim.x;
  sharedMem[tid] = input[offset + tid];
  sharedMem[blockDim.x + tid] = input[offset + blockDim.x + tid];
  for (unsigned int stride = blockDim.x; stride >= 1; stride >>= 1) {
    __syncthreads();
    if ((tid < stride) && ((tid + offset + stride) < len)) {
      sharedMem[tid] += sharedMem[tid+stride];
    }
  }
  
  if (tid == 0) {
    output[blockIdx.x] = sharedMem[0];
  }
  
}


__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  __shared__ volatile float sharedMem[2*BLOCK_SIZE];

  // each thread loads one element from global mem to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int offset = blockIdx.x * (blockDim.x * 2);
  unsigned int tid_global = tid + offset;
  
  // first stage of the reduction on the global to shared memory
  
  // if (tid_global < len){
    sharedMem[tid] = input[tid_global];
    sharedMem[tid + blockDim.x] = input[tid_global + blockDim.x];
  // } else {
  //   sharedMem[tid] = 0;
  // }
  __syncthreads();

  // do reduction in shared mem
  // this loop now starts with s = BLOCK_SIZE 
  // and reduces by factor of 2 every iteration 
  for (unsigned int stride = blockDim.x; stride > 64; stride >>= 1) {
    __syncthreads();
    if (tid < stride && tid_global+stride < len) {
      sharedMem[tid] += sharedMem[tid+stride];
    }
  }

  if (tid < 64){
    sharedMem[tid] += sharedMem[tid + 64];
    sharedMem[tid] += sharedMem[tid + 32];
    sharedMem[tid] += sharedMem[tid + 16];
    sharedMem[tid] += sharedMem[tid + 8];
    sharedMem[tid] += sharedMem[tid + 4];
    sharedMem[tid] += sharedMem[tid + 2];
    sharedMem[tid] += sharedMem[tid + 1];
  }

  // write result for this block to global mem
  if (tid == 0){
    output[blockIdx.x] = sharedMem[0];
  }
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  size_t inputElementsSize = numInputElements*sizeof(float);
  size_t outputElementsSize = numOutputElements*sizeof(float);
  hipMalloc((void **)&deviceInput, inputElementsSize);
  hipMalloc((void **)&deviceOutput, outputElementsSize);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  hipMemcpy(deviceInput, hostInput, inputElementsSize, hipMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(numOutputElements, 1, 1);  

  wbTime_start(Compute, "Performing HIP computation");
  //@@ Launch the GPU Kernel here
  hipLaunchKernelGGL(reduceSmemShfl, dimGrid, dimBlock, 0, 0,
                      deviceInput, deviceOutput, numInputElements);
  
  hipDeviceSynchronize();
  wbTime_stop(Compute, "Performing HIP computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  hipMemcpy(hostOutput, deviceOutput, outputElementsSize, hipMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  hipFree(deviceInput);
  hipFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
