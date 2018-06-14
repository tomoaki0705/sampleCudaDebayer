#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <opencv2/opencv.hpp>
#define KEY_HELP "help"
#define KEY_WIDTH "width"
#define KEY_HEIGHT "height"

const char keys[] = 
    "{" KEY_HELP    " h ?|false|help message}"
    "{" KEY_WIDTH       "|1920 |image width }"
    "{" KEY_HEIGHT      "|1080 |image height}"
    ;

#define CUDA_SAFE_CALL(func) \
do { \
        cudaError_t err = (func); \
        if (err != cudaSuccess) { \
                fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
                exit(err); \
        } \
} while (0)

extern "C" void
launchCudaProcessUnsignedChar(dim3 grid, dim3 block, unsigned char* srcImage, unsigned char* dstImage, int imgW, int imgH);

enum processType
{
    elementWise,
    vector3,
};

template<typename T> void launchCudaProcess(dim3 grid, dim3 block, T* srcImage, T* dstImage, int imgW, int imgH);

template<> void launchCudaProcess(dim3 grid, dim3 block, unsigned char* srcImage, unsigned char* dstImage, int imgW, int imgH)
{
    launchCudaProcessUnsignedChar(grid, block, srcImage, dstImage, imgW, imgH);
}

template<typename T> void launchDebayerPorcess(int W, int H, int gridX, int gridY, enum processType t = elementWise)
{
    std::vector<double> uploadHistory, downloadHistory, processHistory;
    for (size_t iTest = 0; iTest < 20; iTest++)
    {
        const int cElement = W*H;
        const int size = cElement * sizeof(T);
        T* bayerImage, *colorImage, *cpuSrc, *cpuDst;

        // allocate memory
	    CUDA_SAFE_CALL(cudaMalloc((T**)&bayerImage, size));
	    CUDA_SAFE_CALL(cudaMalloc((T**)&colorImage, size*3));
	    cpuSrc = (T*)malloc(size);
	    cpuDst = (T*)malloc(size*3);
        memset(cpuSrc, 0xcc, size/2);
        memset(cpuDst, 0x0, size*3);

	    dim3 block(gridX, gridY, 1); // 16,16
	    dim3 grid(W / block.x * 2, H / block.y * 2, 1);

        int startCount = cv::getTickCount();
        CUDA_SAFE_CALL(cudaMemcpy(bayerImage, cpuSrc, size, cudaMemcpyHostToDevice));
        int startProcess = cv::getTickCount();
        launchCudaProcess<T>(grid, block, bayerImage, colorImage, W, H);
        int processFinished = cv::getTickCount();
        CUDA_SAFE_CALL(cudaMemcpy(cpuDst, colorImage, size * 3, cudaMemcpyDeviceToHost));
        int copyFinished = cv::getTickCount();

	    CUDA_SAFE_CALL(cudaFree((void*)bayerImage));
	    CUDA_SAFE_CALL(cudaFree((void*)colorImage));

        free((void*)cpuSrc);
	    free((void*)cpuDst);
        double tickFrequency = cv::getTickFrequency();
        double copyUpload = ((startProcess - startCount) * 1000) / tickFrequency;
        double processTime = ((processFinished - startProcess) * 1000) / tickFrequency;
        double copyDownload = ((copyFinished - processFinished) * 1000) / tickFrequency;
        uploadHistory.push_back(copyUpload);
        downloadHistory.push_back(copyDownload);
        processHistory.push_back(processTime);
    }
    std::sort(uploadHistory.begin(), uploadHistory.end());
    std::sort(downloadHistory.begin(), downloadHistory.end());
    std::sort(processHistory.begin(), processHistory.end());
    std::cout << W*H << '\t' << uploadHistory[uploadHistory.size() / 2] << '\t' << processHistory[processHistory.size() / 2] << '\t' << downloadHistory[downloadHistory.size()/2] << '\t' << W << 'x' << H << std::endl;
}

int main(int argc, char**argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>(KEY_HELP))
    {
        parser.printMessage();
        return 0;
    }
    launchDebayerPorcess<unsigned char>(parser.get<int>(KEY_WIDTH), parser.get<int>(KEY_HEIGHT), 16, 16);
    return 0;
}
