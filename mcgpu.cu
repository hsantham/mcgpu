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

struct scores {
    double score;
    unsigned int index;
};

typedef struct scores SCORE;

typedef struct termvector TermVector;

__device__ __host__   BigBoy findNC2(int n) {
return ((n * (n-1))/2);
}

// USER DEFINED TYPE DEFs
typedef struct chunkOrder ChunkOrder;

// Global variables
FileName filename, outputFilename;

__device__ __host__  void preprocess(FileContent   fileContent, unsigned int myIdx, unsigned int myLimit) {
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

#define VECTOR_SIZE 1300
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
                   SCORE*        dScores,
                   unsigned int* dSyncBuffer,
                   TermVector*   dTermVector,
                   unsigned int* dUsedArr)
{
    unsigned int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(noOfChunks <= myIdx) return;

    preprocess(fileContent, myIdx * chunkSize, min(fileSize, (myIdx+1) * chunkSize));
    __syncthreads();

    unsigned int used = 0;
    TermVector *myVector = dTermVector + myIdx * TERMS_PER_CHUNK;
    getVector(fileContent + myIdx * chunkSize, 
              min(chunkSize, fileSize - myIdx * chunkSize),
              myVector,
              &used);
    dUsedArr[myIdx] = used;

    if(myIdx == 0) 
        memset(dScores, 0, sizeof(SCORE)*noOfChunks*noOfChunks);
}

__global__ void computeScore(TermVector*   dTermVector,
                             unsigned int* dUsedArr,
                             BigBoy        noOfChunks,
                             SCORE*        dScores) {
    unsigned int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nc2 = findNC2(noOfChunks);
    if(nc2 <= myIdx) return;

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

    double score = getScore(dTermVector + firstChunk * TERMS_PER_CHUNK,
                            dUsedArr[firstChunk],
                            dTermVector + secondChunk * TERMS_PER_CHUNK,
                            dUsedArr[secondChunk]);
    dScores[firstChunk * noOfChunks + secondChunk].score = score;
    dScores[firstChunk * noOfChunks + secondChunk].index = secondChunk;
    dScores[secondChunk * noOfChunks + firstChunk].score = score;
    dScores[secondChunk * noOfChunks + firstChunk].index = firstChunk;
}

__device__ void iSort(SCORE *arr, unsigned int n) {
    int i, j;
    for(i = 1; i < n; i++) {
        SCORE tmp = arr[i];
        for(j = i - 1; j >= 0; j--) {
            if(arr[j].score < tmp.score) {
                break;
            }
            arr[j+1] = arr[j];
        }
        arr[j+1] = tmp;
    }
}

#define sumOfN(n) ((n) *((n)+1))/2

__global__ void sortScores(SCORE*       dScores,
                           BigBoy        noOfChunks) {
    unsigned int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(myIdx >= noOfChunks) return;

    iSort(dScores + myIdx * noOfChunks, noOfChunks);
}
                              
__device__ unsigned int findIndex(SCORE*       dScores,
                                  unsigned int n,
                                  BigBoy        noOfChunks,
                                  unsigned int kthBest) {
    return dScores[n*noOfChunks + kthBest].index;
}

__device__ unsigned int isAlreadyThere(unsigned int *order,
                                       unsigned int n,
                                       unsigned int index) {
    for(unsigned int i = 0; i<n; i++) {
        if(order[i] == index) return 1;
    }

    return 0;
}

__global__ void getOrder(SCORE*       dScores,
                         BigBoy        noOfChunks) {
    unsigned int *order = (unsigned int *) malloc(sizeof(unsigned int) * noOfChunks);
    unsigned int i, k;
    order[0] = 0;
    for(i=1; i<noOfChunks; i++) {
        k=0;
        while(1) {
            unsigned int index = findIndex(dScores, order[i-1], noOfChunks, k);
            if(!isAlreadyThere(order, i, index)) {
                order[i] = index;
                break;
            }
            k++;
        }
    }

    for(i = 0; i < noOfChunks; i++) {
        dScores[i].score = order[i];
        dScores[i].index = order[i];
    }
}

size_t getFilesize(FileName filename) {
    struct stat st;
    if(stat((const char *)filename, &st) != 0) {
        assert(0);
        return 0;
    }
    return st.st_size;   
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
    outputFilename = (unsigned char *) "/home/anand/Desktop/hemanth/phase2/mcgpu/otext8";

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
    SCORE *hScores;
    hostFileBuffer = (FileContent) malloc(filesize+1);
    reorderedFileBuffer = (FileContent) malloc(filesize+1);
    getFileContent(filename, hostFileBuffer);

    hostReorderInfo = (ChunkOrder *) malloc(sizeof(ChunkOrder)*noOfChunks);
    hScores = (SCORE *) malloc(sizeof(SCORE)*noOfChunks*noOfChunks);
    assert(hScores != NULL);

    ChunkOrder *deviceReorderInfo;
    SCORE *dScores;
    unsigned int *dSyncBuffer, *dUsedArr;
    TermVector *dVector;
    checkCudaErrors(cudaMalloc((void **)&deviceFileBuffer, filesize));
    checkCudaErrors(cudaMalloc((void **)&dScores, sizeof(SCORE)*noOfChunks*noOfChunks));
    checkCudaErrors(cudaMalloc((void **)&dSyncBuffer, sizeof(unsigned int)*10));
    checkCudaErrors(cudaMalloc((void **)&dVector, noOfChunks * TERMS_PER_CHUNK * sizeof(TermVector)));
    checkCudaErrors(cudaMalloc((void **)&dUsedArr, noOfChunks * sizeof(unsigned int)));

    checkCudaErrors(cudaMemcpy(deviceFileBuffer, hostFileBuffer, filesize, cudaMemcpyHostToDevice));

    printf("Launching CUDA kernal\n");

    noOfBlocks = ceil((double)noOfChunks/(double)32);
    threadsPerBlock = 32;
    sq<<<noOfBlocks, threadsPerBlock>>>(deviceFileBuffer, 
                                        filesize, 
                                        noOfChunks, 
                                        chunkSize, 
                                        dScores, 
                                        dSyncBuffer, 
                                        dVector,
                                        dUsedArr);
    
    noOfBlocks = ceil((double)nc2/(double)256);;
    threadsPerBlock = 256;
    computeScore<<<noOfBlocks, threadsPerBlock>>>(dVector, dUsedArr, noOfChunks, dScores);

    noOfBlocks = ceil((double)noOfChunks/(double)32);
    threadsPerBlock = 32;
    sortScores<<<noOfBlocks, threadsPerBlock>>>(dScores, noOfChunks);

    getOrder<<<1,1>>>(dScores, noOfChunks);

    checkCudaErrors(cudaMemcpy(hScores, 
                               dScores, 
                               sizeof(SCORE)*noOfChunks, 
                               cudaMemcpyDeviceToHost));
    printf("CUDA kernel execution over !!!\n");

    checkCudaErrors(cudaFree(deviceFileBuffer));
    checkCudaErrors(cudaFree(dScores));
    checkCudaErrors(cudaFree(dSyncBuffer));

/*    hScores[0] = getScore(hostFileBuffer, 
                            chunkSize, 
                            hostFileBuffer + 1*chunkSize, 
                            min(chunkSize, filesize - chunkSize));
*/

    unsigned int i, j;
    for(i = 0; i < noOfChunks; i++) {
        for(j = 0; j < noOfChunks; j++) {
            printf("%.4lf,%u",hScores[i*noOfChunks + j].score, hScores[i*noOfChunks + j].index);
            printf(" ");
        }
        printf("\n");
        break;
    }

    // open file descriptor
    FILE *file;
    file = fopen((const char *)outputFilename, "w");
    assert(file != NULL);

    size_t written;
    for(i = 0; i < noOfChunks; i++) {
        written = fwrite(hostFileBuffer + hScores[i].index * chunkSize, 1, chunkSize, file);
        assert(written == chunkSize);
        assert(ferror(file) == 0);
    }

    fclose(file);

    return 0;
}
