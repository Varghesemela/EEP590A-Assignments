#include <hip/hip_runtime.h>
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i<len) out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  hipMalloc((void **)&deviceInput1, inputLength * sizeof(float));
  hipMalloc((void **)&deviceInput2, inputLength * sizeof(float));
  hipMalloc((void **)&deviceOutput, inputLength * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  hipStream_t stream1;
  hipStream_t stream2;
  hipStreamCreate(&stream1);
  hipStreamCreate(&stream2);
  hipMemcpyAsync(deviceInput1, hostInput1, inputLength*sizeof(float), hipMemcpyHostToDevice, stream1);
  hipMemcpyAsync(deviceInput2, hostInput2, inputLength*sizeof(float), hipMemcpyHostToDevice, stream2);
  hipDeviceSynchronize();
  hipStreamDestroy(stream1);
  hipStreamDestroy(stream2);
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 numthreads(256, 1, 1);
  dim3 numblocks((inputLength+256-1)/256, 1, 1);

  wbTime_start(Compute, "Performing HIP computation");
  //@@ Launch the GPU Kernel here
  hipLaunchKernelGGL(vecAdd, numblocks, numthreads, 0, 0, deviceInput1, deviceInput2, deviceOutput, inputLength);
  hipDeviceSynchronize();
  wbTime_stop(Compute, "Performing HIP computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  hipMemcpy(hostOutput, deviceOutput, inputLength*sizeof(float), hipMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  hipFree(deviceInput1);
  hipFree(deviceInput2);
  hipFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
