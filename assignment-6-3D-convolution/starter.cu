#include <wb.h>
#include <hip/hip_runtime.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "HIP error: ", hipGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_RADIUS 1
#define MASK_DIM ((MASK_RADIUS) * 2 + 1)
#define STRIDE 1

//@@ Define constant memory for device kernel here
__constant__ float mask_c[MASK_DIM][MASK_DIM][MASK_DIM];

__global__ void conv3d(float *input, float *output, const int z_size, const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  int tid_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (tid_x < x_size && tid_y < y_size && tid_z < z_size) {
      float sum = 0.0f;

      for(int mask_X = 0; mask_X < MASK_DIM; ++mask_X) {
        for(int mask_Y = 0; mask_Y < MASK_DIM; ++mask_Y) {
          for(int mask_Z = 0; mask_Z < MASK_DIM; ++mask_Z) {
            int input_x = tid_x - MASK_RADIUS + mask_X;
            int input_y = tid_y - MASK_RADIUS + mask_Y;
            int input_z = tid_z - MASK_RADIUS + mask_Z;
            if(input_x >= 0 && input_x < x_size && input_y >= 0 && input_y < y_size && input_z >= 0 && input_z < z_size) {
              sum +=  input[(input_z * y_size * x_size) + (input_y * x_size) + input_x] * mask_c[mask_X][mask_Y][mask_Z];
            }
          }
        }
      }
      output[(tid_z * y_size * x_size) + (tid_y * x_size) + tid_x] = sum;
  }
}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbCheck(hipMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float)));
  wbCheck(hipMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float)));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbCheck(hipMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), hipMemcpyHostToDevice));
  // Copying the kernel to constant memory
  wbCheck(hipMemcpyToSymbol(mask_c, hostKernel, MASK_DIM * MASK_DIM * MASK_DIM * sizeof(float)));
  wbCheck(hipMemcpy(deviceOutput, hostOutput + 3, (inputLength - 3) * sizeof(float), hipMemcpyHostToDevice));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(8, 8, 8);
  dim3 dimGrid((x_size + dimBlock.x - 1) / dimBlock.x, (y_size + dimBlock.y - 1) / dimBlock.y, (z_size + dimBlock.z - 1) / dimBlock.z);
  
  //@@ Launch the GPU kernel here
  hipLaunchKernelGGL(conv3d, dimGrid, dimBlock, 0, 0, 
                      deviceInput, deviceOutput, 
                      z_size, y_size, x_size
                    );
  hipDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbCheck(hipMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), hipMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  hipFree(deviceInput);
  hipFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}