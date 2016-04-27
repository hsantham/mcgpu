//extern "C"

#include<stdio.h>
#include <sys/stat.h>
#include <assert.h>
#include <stdint.h>

#define TERMS_PER_CHUNK 1000

// TYPE DEFs
typedef unsigned char * FileName;
typedef unsigned char * FileContent;
typedef unsigned long long int BigBoy;

// User defined data types
struct chunkOrder {
  BigBoy chunkOffset;
  BigBoy reorderedPosition;
};

struct termvector {
    unsigned char *term;
    unsigned short count;
};

typedef struct termvector TermVector;

__device__ __host__   BigBoy findNC2(int n) {
return ((n * (n-1))/2);
}

// USER DEFINED TYPE DEFs
typedef struct chunkOrder ChunkOrder;

// Global variables
FileName filename;

__device__ __host__  void preprocess(FileContent   fileContent, int myIdx, int myLimit) {
    for(unsigned int i = myIdx; i < myLimit; i++) {
        switch(fileContent[i]) {
            case '.':
            case '(':
            case ')':
            case ':':
            case '-':
            case ',':
                fileContent[i] = ' ';
                break;
        }
    }
}

__device__ unsigned int isMatch(unsigned char *term1, unsigned char *term2) {
    while(*term1 == *term2) {
        if(*term1 == '\0' || *term1 == ' ') return 1;
        term1++; term2++;
    }
    
    return 0;
}


void printVector(TermVector *vector, unsigned int used) {
    unsigned int i;
    for(i = 0; i < used; i++) {
        unsigned int j=0;
        while(vector[i].term[j] != ' ' && vector[i].term[j] != '\0') {
            printf("%c", vector[i].term[j]);
            j++;
        }
        printf("   = %d\n",vector[i].count);
    }
}

__device__ unsigned int findIndex(TermVector *vector1, unsigned int vector1Count, FileContent term) {
    unsigned int j;
    for(j = 0; j < vector1Count; j++) {
        if(isMatch(vector1[j].term, term)) {
            return j;
        }
    }

    return (unsigned int) -1;
}

#define VECTOR_SIZE 1000
__device__ void getVector(FileContent   content1, 
                unsigned int  letterCount1,
                TermVector *vector1,
                unsigned int *used) {
    unsigned int i = 0, startIdx = (unsigned int)-1;
    for(; i < letterCount1; i++) {
        if(content1[i] != ' ') {
            startIdx = (startIdx == (unsigned int)-1) ? i : startIdx;
            continue;
        }

        if(startIdx == (unsigned int)-1) {
            continue;
        }


        unsigned int index = findIndex(vector1, *used, content1+startIdx);
        if(index == (unsigned int)-1) {
            assert(*used < VECTOR_SIZE);
            vector1[*used].term = content1+startIdx;
            vector1[*used].count = 1;
            (*used)++;
        } else {
            vector1[index].count++;
        }
        startIdx = (unsigned int)-1;
    }

    if(startIdx != (unsigned int)-1) {
        unsigned int index = findIndex(vector1, *used, content1+startIdx);
        if(index == (unsigned int)-1) {
            assert(*used < VECTOR_SIZE);
            vector1[*used].term = content1+startIdx;
            vector1[*used].count = 1;
            (*used)++;
        } else {
            vector1[index].count++;
        }
        
    }
}

__device__ unsigned int getADotB(TermVector *vector1, 
                      unsigned int vector1Count,
                      TermVector *vector2,
                      unsigned int vector2Count) {
    unsigned int i;
    unsigned int aDotb = 0;
    for(i = 0; i < vector1Count; i++) {
        unsigned int index = findIndex(vector2, vector2Count, vector1[i].term);
        if(index != (unsigned int)-1) {
            aDotb += vector1[i].count * vector2[index].count;
        }
    }

    return aDotb;
}

__device__ double modOfVector(TermVector *vector, unsigned int vectorCount) {
    unsigned int i;
    unsigned int modValue = 0;
    for(i = 0; i < vectorCount; i++) {
        modValue += vector[i].count * vector[i].count;
    }

    return sqrt((double) modValue);
}

__device__ double getScore(TermVector *vector1, 
                           unsigned int vector1Count,
                           TermVector *vector2,
                           unsigned int vector2Count) {
    unsigned int dotProduct = getADotB(vector1, vector1Count, vector2, vector2Count); 
    double magA = modOfVector(vector1, vector1Count);
    double magB = modOfVector(vector2, vector2Count);
    return (double)dotProduct/(magA * magB);
}

__global__ void sq(FileContent   fileContent, 
                   BigBoy        fileSize, 
                   BigBoy        noOfChunks,
                   BigBoy        chunkSize,
                   double*       dScores,
                   unsigned int* dSyncBuffer,
                   TermVector*   dTermVector,
                   unsigned int* dUsedArr)
{
    dSyncBuffer[0] = 0;
    __syncthreads();

    int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int myLimit = myIdx + chunkSize;
    int nc2 = findNC2(noOfChunks);
    if(nc2 <= myIdx) return;

    preprocess(fileContent, myIdx, myLimit);
    __syncthreads();

    unsigned int firstChunk, secondChunk, i, j, k = 0; 
    for(i = 0; i <= (noOfChunks-1-1); i++) {
        for(j = i+1; j <= (noOfChunks-1); j++) {
            if(myIdx == k) {
                firstChunk = i;
                secondChunk = j;
                i = noOfChunks; // To break from outer for loop
                break;
            }
            k++;
        }
    } 

    __syncthreads();

    if(myIdx < noOfChunks) {
        unsigned int used = 0;
        TermVector *myVector = dTermVector + myIdx * TERMS_PER_CHUNK;
        getVector(fileContent + myIdx*chunkSize, 
                  min(chunkSize,fileSize-myIdx*chunkSize),
                  myVector,
                  &used);
        dUsedArr[myIdx] = used;
        atomicAdd(&dSyncBuffer[0], 1);
    }

    while(dSyncBuffer[0] == noOfChunks);

    double score = getScore(dTermVector + firstChunk * TERMS_PER_CHUNK,
                            dUsedArr[firstChunk],
                            dTermVector + secondChunk * TERMS_PER_CHUNK,
                            dUsedArr[secondChunk]);
    dScores[myIdx] = score;
}

size_t getFilesize(FileName filename) {
    struct stat st;
    if(stat((const char *)filename, &st) != 0) {
        assert(0);
        return 0;
    }
    return st.st_size - 1;   
}

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

int main(int argc, char *argv[]) {

    filename = (unsigned char *) "/home/anand/Desktop/hemanth/phase2/mcgpu/text8";

    FileContent deviceFileBuffer, hostFileBuffer, reorderedFileBuffer;
    BigBoy filesize = getFilesize(filename);
    printf("%llu\n\n", filesize);    

    BigBoy chunkSize = (argc == 1) ? 1024 : atoi(argv[1]);
    BigBoy noOfChunks = (int) ceil(((double)filesize/(double)chunkSize));
    BigBoy nc2 = findNC2(noOfChunks);
    unsigned int threadsPerBlock = 1024;
    unsigned int noOfBlocks = (int) ceil(((double)nc2/(double)threadsPerBlock));

    printf("File size = %llu; chunk size = %llu;    no of chunks = %llu;\n", filesize, chunkSize, noOfChunks);
    printf("nc2 = %llu;       threadsPerBlock = %u; no of Blocks = %u; \n", nc2, threadsPerBlock, noOfBlocks);

    ChunkOrder *hostReorderInfo;
    double *hScores;
    hostFileBuffer = (FileContent) malloc(filesize+1);
    reorderedFileBuffer = (FileContent) malloc(filesize+1);
    getFileContent(filename, hostFileBuffer);

    hostReorderInfo = (ChunkOrder *) malloc(sizeof(ChunkOrder)*noOfChunks);
    hScores = (double *) malloc(sizeof(double)*noOfChunks*noOfChunks);

    ChunkOrder *deviceReorderInfo;
    double *dScores;
    unsigned int *dSyncBuffer, *dUsedArr;
    TermVector *dVector;
    checkCudaErrors(cudaMalloc((void **)&deviceFileBuffer, filesize));
    checkCudaErrors(cudaMalloc((void **)&dScores, sizeof(double)*nc2));
    checkCudaErrors(cudaMalloc((void **)&dSyncBuffer, sizeof(unsigned int)*10));
    checkCudaErrors(cudaMalloc((void **)&dVector, noOfChunks * TERMS_PER_CHUNK * sizeof(TermVector)));
    checkCudaErrors(cudaMalloc((void **)&dUsedArr, noOfChunks * sizeof(unsigned int)));

    checkCudaErrors(cudaMemcpy(deviceFileBuffer, hostFileBuffer, filesize, cudaMemcpyHostToDevice));

    printf("Launching CUDA kernal\n");

    sq<<<noOfBlocks, threadsPerBlock>>>(deviceFileBuffer, 
                                        filesize, 
                                        noOfChunks, 
                                        chunkSize, 
                                        dScores, 
                                        dSyncBuffer, 
                                        dVector,
                                        dUsedArr);

    checkCudaErrors(cudaMemcpy(hScores, dScores, sizeof(double)*nc2, cudaMemcpyDeviceToHost));
    printf("CUDA kernel execution over !!!\n");

    checkCudaErrors(cudaFree(deviceFileBuffer));
    checkCudaErrors(cudaFree(dScores));
    checkCudaErrors(cudaFree(dSyncBuffer));

/*    hScores[0] = getScore(hostFileBuffer, 
                            chunkSize, 
                            hostFileBuffer + 1*chunkSize, 
                            min(chunkSize, filesize - chunkSize));
*/

    unsigned int i;
    for(i = 0; i < nc2; i++) {
        printf("%f\n",hScores[i]);
    }

    return 0;
}
