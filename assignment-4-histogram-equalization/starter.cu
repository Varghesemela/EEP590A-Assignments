// Histogram Equalization
#include<hip/hip_runtime.h>
#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE  32

//@@ insert code here

__global__ void FloatToUChar(float *floatImage, unsigned char *ucharImage, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < height && col < width) {
    int index = blockIdx.z * (width * height) + row * width + col;
    ucharImage[index] = (unsigned char) (255 * floatImage[index]);
  }
}

__global__ void RGBtoGray(unsigned char *rgbImage, unsigned char *grayImage, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < height && col < width) {
    int index = row * width + col;
    unsigned char redChannel = rgbImage[3 * index];
    unsigned char greenChannel = rgbImage[3 * index + 1];
    unsigned char blueChannel = rgbImage[3 * index + 2];
    grayImage[index] = (unsigned char) (0.21*redChannel + 0.71*greenChannel + 0.07*blueChannel);
  }
}

__global__ void calcHistogram(unsigned char *Image, unsigned int *histogram, int width, int height) {
  int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (tid_global < HISTOGRAM_LENGTH) {
    histogram[tid_global] = 0;
  }  
  __syncthreads();

  while (tid_global < width * height) {
    atomicAdd( &(histogram[Image[tid_global]]), 1);
    tid_global += stride;
  }
}

__global__ void calcCDF(unsigned int *histogram, float *CDFdata, int width, int height){
  __shared__ unsigned int sharedMem[HISTOGRAM_LENGTH];
  int tid = threadIdx.x;
  int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
  // each thread loads one element from global mem to shared mem
  if (tid_global < HISTOGRAM_LENGTH) {
    sharedMem[tid] = histogram[tid_global];
  }
  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
    __syncthreads();
    if (tid >= stride) {
      sharedMem[tid] += sharedMem[tid - stride];
    }
  }
  __syncthreads();

  if (tid_global < HISTOGRAM_LENGTH){
    CDFdata[tid_global] = sharedMem[tid_global] / ((float)(width * height));
  }
}

__global__ void equalize(unsigned char *Image, float *CDFdata, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < height && col < width) {
    int index = blockIdx.z * (width * height) + row * width + col;
    float cdfmin = CDFdata[0];
    Image[index] = (unsigned char) min(max(255*(CDFdata[Image[index]] - cdfmin)/(1.0 - cdfmin), 0.0), 255.0);
  }
}

__global__ void UCharToFloat(unsigned char *ucharImage, float *floatImage, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < height && col < width) {
    int index = blockIdx.z * (width * height) + row * width + col;
    floatImage[index] = (float) (ucharImage[index]/255.0);
  }
}


//@@ insert code here


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *d_InputImageData;
  unsigned char *d_ImageChar; // RGB
  unsigned char *d_ImageGray;  // Gray scale
  unsigned int  *d_Histogram;  // Histogram
  float         *d_ImageCDF;        // CDF
  float         *d_OutputImageData;     // Output

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");


  //@@ insert code here
  int imageResolution = imageWidth * imageHeight;
  int imageSize = imageResolution * imageChannels;

  hipMalloc((void **) &d_InputImageData, imageSize * sizeof(float));
  hipMalloc((void **) &d_ImageChar, imageSize * sizeof(unsigned char));
  hipMalloc((void **) &d_ImageGray, imageResolution * sizeof(unsigned char));
  hipMalloc((void **) &d_Histogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  hipMalloc((void **) &d_ImageCDF, HISTOGRAM_LENGTH * sizeof(unsigned int));
  hipMalloc((void **) &d_OutputImageData, imageSize * sizeof(float));

  hipMemcpy(d_InputImageData, hostInputImageData, imageSize * sizeof(float), hipMemcpyHostToDevice);

  dim3 dimGrid_RGB(ceil((1.0 * imageWidth)/BLOCK_SIZE), ceil((1.0 * imageHeight)/BLOCK_SIZE), imageChannels);
  dim3 dimBlock_Tile(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 dimGrid_Gray(ceil((1.0 * imageWidth)/BLOCK_SIZE), ceil((1.0 * imageHeight)/BLOCK_SIZE), 1);
  dim3 dimGrid_Histo(1, 1, 1);
  dim3 dimBlock_Histo(HISTOGRAM_LENGTH, 1, 1);

  hipLaunchKernelGGL(FloatToUChar, dimGrid_RGB, dimBlock_Tile, 0, 0,
                        d_InputImageData, d_ImageChar,
                        imageWidth, imageHeight);
  hipDeviceSynchronize();
  
  hipLaunchKernelGGL(RGBtoGray, dimGrid_Gray, dimBlock_Tile, 0, 0,
                        d_ImageChar, d_ImageGray, 
                        imageWidth, imageHeight);
  hipDeviceSynchronize();

  hipLaunchKernelGGL(calcHistogram, dimGrid_Histo, dimBlock_Histo, 0, 0,
                        d_ImageGray, d_Histogram, 
                        imageWidth, imageHeight);
  hipDeviceSynchronize();

  hipLaunchKernelGGL(calcCDF, dimGrid_Histo, dimBlock_Histo, 0, 0,
                        d_Histogram, d_ImageCDF, 
                        imageWidth, imageHeight);
  hipDeviceSynchronize();

  hipLaunchKernelGGL(equalize, dimGrid_RGB, dimBlock_Tile, 0, 0,
                        d_ImageChar, d_ImageCDF, 
                        imageWidth, imageHeight);
  hipDeviceSynchronize();

  hipLaunchKernelGGL(UCharToFloat, dimGrid_RGB, dimBlock_Tile, 0, 0, 
                        d_ImageChar, d_OutputImageData, 
                        imageWidth, imageHeight);
  hipDeviceSynchronize();

  hipMemcpy(hostOutputImageData, d_OutputImageData, imageSize * sizeof(float), hipMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  // Free GPU Memory 
  hipFree(d_InputImageData);
  hipFree(d_ImageChar);
  hipFree(d_ImageGray);
  hipFree(d_Histogram);
  hipFree(d_ImageCDF);
  hipFree(d_OutputImageData);
  // Free CPU Memory
  free(hostInputImageData);
  free(hostOutputImageData);
  return 0;
}
