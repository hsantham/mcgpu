extern "C"
{
#include<stdio.h>
#include <sys/stat.h>
#include <assert.h>
}

// TYPE DEFs
typedef unsigned char * FileName;
typedef unsigned char * FileContent;
typedef unsigned long long int BigBoy;

// User defined data types
struct chunkOrder {
  BigBoy chunkOffset;
  BigBoy reorderedPosition;
};

// USER DEFINED TYPE DEFs
typedef struct chunkOrder ChunkOrder;

// Global variables
FileName filename;

extern "C"
__global__ void sq(FileContent   fileContent, 
                   BigBoy        fileSize, 
                   BigBoy        n,
                   BigBoy        chunkSize,
                   ChunkOrder    *reorderInfo)
{
    int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int myLimit = myIdx + chunkSize;

    FileContent myContent = fileContent + myIdx;
    *myContent = '0' + myIdx/10;
    
    int summa;
    summa = 10;
}

extern "C"
BigBoy findNC2(int n) {
return ((n * (n-1))/2);
}

extern "C"
size_t getFilesize(FileName filename) {
    struct stat st;
    if(stat((const char *)filename, &st) != 0) {
        return 0;
    }
    return st.st_size;   
}

extern "C"
void getFileContent(FileName filename, FileContent buffer) {
FILE *file;
size_t nread;

file = fopen((const char *)filename, "r");
assert(file != NULL);

BigBoy filesize = getFilesize(filename);

nread = fread(buffer, 1, filesize, file);
assert(nread == filesize);
assert(ferror(file) == 0);

fclose(file);
}

#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(int err, const char *file, const int line)
{
    if (0 != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d  from file <%s>, line %i.\n",
                err, file, line);
        exit(EXIT_FAILURE);
    }
}
#endif

extern "C"
int main() {

filename = (unsigned char *) "/home/anand/Desktop/hemanth/phase2/text8";

FileContent deviceFileBuffer, hostFileBuffer;
BigBoy filesize = getFilesize(filename);
BigBoy chunkSize = 1024*1024;
ChunkOrder *deviceReorderInfo, *hostReorderInfo;
BigBoy n = (int) ceil(((double)filesize/(double)chunkSize));
BigBoy nc2 = findNC2(n);

hostFileBuffer = (FileContent) malloc(filesize+1);
getFileContent(filename, hostFileBuffer);

checkCudaErrors(cudaMalloc((void **)&deviceFileBuffer, filesize));
checkCudaErrors(cudaMalloc((void **)&deviceReorderInfo, sizeof(ChunkOrder)*n));
hostReorderInfo = (ChunkOrder *) malloc(sizeof(ChunkOrder)*n);


checkCudaErrors(cudaMemcpy(deviceFileBuffer, hostFileBuffer, filesize, cudaMemcpyHostToDevice));

sq<<<1, 1>>>(deviceFileBuffer, filesize, n, chunkSize, deviceReorderInfo);
//sq<<<nc2/1024, 1024>>>(deviceFileBuffer, filesize, n, chunkSize, deviceReorderInfo);

checkCudaErrors(cudaMemcpy(hostReorderInfo, deviceReorderInfo, sizeof(ChunkOrder)*n, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(hostFileBuffer, deviceFileBuffer, filesize, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaFree(deviceReorderInfo));
checkCudaErrors(cudaFree(deviceFileBuffer));
return 0;
}
