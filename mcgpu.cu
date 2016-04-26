//extern "C"

#include<stdio.h>
#include <sys/stat.h>
#include <assert.h>
#include <stdint.h>


// TYPE DEFs
typedef unsigned char * FileName;
typedef unsigned char * FileContent;
typedef unsigned long long int BigBoy;

__device__ float llAllocCount;

// User defined data types
struct chunkOrder {
  BigBoy chunkOffset;
  BigBoy reorderedPosition;
};

struct llnode {
    void *data;
    struct llnode *next;
};

typedef struct llnode LLNode;

struct termvector {
    unsigned char *term;
    unsigned short count;
};

typedef struct termvector TermVector;
struct hnode {
    unsigned char *string;
    unsigned char source;
    unsigned char count;
};
typedef struct hnode HNode;

//extern "C" 
__device__ __host__  LLNode *getNewLLNode(void *data) {
    //llAllocCount++;
    LLNode *head = (LLNode *) malloc(sizeof(LLNode));
    assert(head != NULL);
    head->data = data;
    head->next = NULL;

    return head;
}

//extern "C" 
__device__ __host__  void append(LLNode *head, LLNode *nodeToInsert) {
  LLNode *node = head;

  while(node->next) node=node->next;

  node->next = nodeToInsert;
  nodeToInsert->next = NULL;
}

//extern "C"
__device__ __host__   BigBoy findNC2(int n) {
return ((n * (n-1))/2);
}

// USER DEFINED TYPE DEFs
typedef struct chunkOrder ChunkOrder;

// Global variables
FileName filename;

//extern "C"
__device__ __host__  void preprocess(FileContent   fileContent, int myIdx, int myLimit) {
    for(int i = myIdx; i < myLimit; i++) {
        switch(fileContent[i]) {
            case '.':
            case '(':
            case ')':
            case ':':
            case '-':
            case ',':
                fileContent[myIdx] = ' ';
                break;
        }
    }
}

#define MAX_HASH_TABLE_ENTRIES 20
#define MAX_NODES_IN_HASH_TABLE 250

//extern "C" 
__device__ __host__  unsigned int getHash(FileContent string, unsigned int length) {
    unsigned int hash = 0;
    for(unsigned int i = 0; i < length; i++) {
        hash = hash + string[i];
    }

    return hash % MAX_HASH_TABLE_ENTRIES;
}

//extern "C" 
__device__ __host__  HNode* getHashNode(FileContent string, 
                   uint8_t     source, 
                   uint8_t     count, 
                   HNode*      nodes, 
                   unsigned int*   used) {
    HNode* node = nodes + *used;
    node->string = string;
    node->source = source;
    node->count  = count;

    *(used)++;
    return node;
} 

//extern "C" 
__device__ __host__  void fillHashTable(LLNode **hashTable, unsigned int hash, LLNode *node) {
    LLNode *head = hashTable[hash];
    if(head == NULL) {
        hashTable[hash] = head;
    } else {
        while(head->next) head = head->next;
        head->next = node;
    }
}

//extern "C" 
__device__ __host__  LLNode* findMatchingNode(LLNode* head, FileContent content) {
    LLNode *node = head;
    while(node) {
        FileContent existing = (FileContent) node->data;
        for(uint64_t i = 0;; i++) {
            if(existing[i] == content[i]) {
                continue;
            }
            
            if(existing[i] == ' ' && content[i] == ' ') {
                return node;
            }

            if(existing[i] == '\0' && content[i] == '\0') {
                return node;
            }

            break;
        }
    }

    return NULL;
}

//extern "C"
 __device__ __host__  double getScoreFromHashTable(LLNode **hashTable) {
    double euclidDist = 0;
    for(unsigned int i = 0; i < MAX_HASH_TABLE_ENTRIES; i++) {
        LLNode *node = hashTable[i];
        while(node) {
            HNode *hnode = (HNode *) node->data;            
            euclidDist += hnode->count * hnode->count;
            node = node->next;
        }
    }

    return sqrt((double) euclidDist);
}


unsigned int isMatch(unsigned char *term1, unsigned char *term2) {
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

unsigned int findIndex(TermVector *vector1, unsigned int vector1Count, FileContent term) {
    unsigned int j;
    for(j = 0; j < vector1Count; j++) {
        if(isMatch(vector1[j].term, term)) {
            return j;
        }
    }

    return (unsigned int) -1;
}

#define VECTOR_SIZE 1000
void getVector(FileContent   content1, 
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

    if(startIdx != -1) {
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

unsigned int getADotB(TermVector *vector1, 
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

double modOfVector(TermVector *vector, unsigned int vectorCount) {
    unsigned int i;
    unsigned int modValue = 0;
    for(i = 0; i < vectorCount; i++) {
        modValue += vector[i].count * vector[i].count;
    }

    return sqrt(modValue);
}

double ggetScore(FileContent   content1, 
                           unsigned int  letterCount1, 
                           FileContent   content2, 
                           unsigned int  letterCount2) {
    TermVector vector[VECTOR_SIZE]={}; 
    unsigned int used = 0, vector1Count, vector2Count;

    TermVector *vector1 = vector;
    getVector(content1, letterCount1, vector, &used);
    printVector(vector1, used);
    vector1Count = used;

    printf("==================================================\n\n\n");

    used = 0;
    TermVector *vector2 = vector + vector1Count;
    getVector(content2, letterCount2, vector2, &used);
    vector2Count = used;
    printVector(vector2, vector2Count);

    assert((vector1Count + vector2Count) < VECTOR_SIZE);

    unsigned int dotProduct = getADotB(vector1, vector1Count, vector2, vector2Count); 
    double magA = modOfVector(vector1, vector1Count);
    double magB = modOfVector(vector2, vector2Count);
    return (double)dotProduct/(magA * magB);
}

//extern "C"
__device__ 
__host__  double getScore(FileContent   content1, 
                           unsigned int  letterCount1, 
                           FileContent   content2, 
                           unsigned int  letterCount2) {
    LLNode *hashTable[MAX_HASH_TABLE_ENTRIES] = {};
    
    int i = 0, startIdx=-1;
    HNode nodes[MAX_NODES_IN_HASH_TABLE]; unsigned int used = 0;
    unsigned char currentSrc = 1;
    for(;i<letterCount1;i++) {
        if(content1[i] != ' ') {
            startIdx = (startIdx == -1) ? i : startIdx;
            continue;
        }

        if(startIdx == -1) {
            continue;
        }

        unsigned char hash = getHash(content1+startIdx, i-startIdx);
        LLNode *head = hashTable[hash];
        
        if(head == NULL) {
            HNode *hashNode = getHashNode(content1+startIdx, currentSrc, 1, nodes, &used);
            LLNode *head = getNewLLNode(hashNode);
            fillHashTable(hashTable, hash, head);

            startIdx = -1;
            continue;
        }

        LLNode *matchingNode = findMatchingNode(head, content1+startIdx);
        if(matchingNode) {
            HNode *hashNode = (HNode *) matchingNode->data;
            hashNode->count++;
            startIdx = -1;
            continue;
        }

        HNode *hashNode = getHashNode(content1+startIdx, currentSrc, 1, nodes, &used);
        LLNode *nodeToInsert = getNewLLNode(hashNode);
        fillHashTable(hashTable, hash, nodeToInsert);

        startIdx = -1;
        continue;
    }

    i = 0; startIdx=-1;
    currentSrc = 2;
    for(;i<letterCount2; i++) {
        if(content2[i] != ' ') {
            startIdx = (startIdx == -1) ? i : startIdx;
            continue;
        }

        if(startIdx == -1) {
            continue;
        }

        unsigned char hash = getHash(content2+startIdx, i-startIdx);
        LLNode *head = hashTable[hash];

        if(head == NULL) {
            HNode *hashNode = getHashNode(content1+startIdx, currentSrc, 1, nodes, &used);
            LLNode *head = getNewLLNode(hashNode);
            fillHashTable(hashTable, hash, head);

            startIdx = -1;
            continue;
        }

        LLNode *matchingNode = findMatchingNode(head, content1+startIdx);
        if(matchingNode) {
            HNode *hashNode = (HNode *) matchingNode->data;
            if(hashNode->source != currentSrc) {
                hashNode->count--;
            } else {
                hashNode->count++;
            }
            startIdx = -1;
            continue;
        }

        HNode *hashNode = getHashNode(content1+startIdx, currentSrc, 1, nodes, &used);
        LLNode *nodeToInsert = getNewLLNode(hashNode);
        fillHashTable(hashTable, hash, nodeToInsert);

        startIdx = -1;
        continue;
    }

    return getScoreFromHashTable(hashTable);
}

//extern "C"
__global__ void sq(FileContent   fileContent, 
                   BigBoy        fileSize, 
                   BigBoy        n,
                   BigBoy        chunkSize,
                   double*       dScores,
                   uint8_t*      dSyncBuffer)
{
    dSyncBuffer[blockIdx.x] = 0;
    __syncthreads();

    int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
//    if(myIdx==0)
//        *llAllocCount=0;
    int myLimit = myIdx + chunkSize;
    int nc2 = findNC2(n);
    if(nc2 <= myIdx) return;

    preprocess(fileContent, myIdx, myLimit);
    __syncthreads();

    unsigned int firstChunk, secondChunk, i, j, k = 0; 
    for(i = 0; i <= (n-1-1); i++) {
        for(j = i+1; j <= (n-1); j++) {
            if(myIdx == k) {
                firstChunk = i;
                secondChunk = j;
                i = n; // To break from outer for loop
                break;
            }
            k++;
        }
    } 

    __syncthreads();

    double score = getScore(fileContent + firstChunk*chunkSize, 
                            chunkSize, 
                            fileContent + secondChunk*chunkSize, 
                            chunkSize);
    dScores[myIdx] = score;

    __syncthreads();
/*    dSyncBuffer[blockIdx.x] = 1;

    unsigned int noOfBlocks = (int) ceil(((double)nc2/(double)threadsPerBlock));
    for(i = 0; i < noOfBlocks; i++) {
        if(dSyncBuffer[i] == 0) i--;
    }

    if(myIdx >= n) return;
    unsigned int start = (myIdx*n - (myIdx*(myIdx+1))/2);
    unsigned int end   = ((myIdx+1)*n - ((myIdx+1)*(myIdx+2))/2)
    unsigned int max = dScores[start], maxIndex = start;
    for(i = start; i < end; i++) {
       if(dScores[i] > max) {
            max = dScores[i];
            maxIndex = i;
       } 
    }
    
    dScores[start] = maxIndex; */
}

//extern "C"
size_t getFilesize(FileName filename) {
    struct stat st;
    if(stat((const char *)filename, &st) != 0) {
        return 0;
    }
    return st.st_size;   
}

//extern "C"
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

//extern "C"
int main(int argc, char *argv[]) {

    filename = (unsigned char *) "/home/anand/Desktop/hemanth/phase2/mcgpu/text8";

    FileContent deviceFileBuffer, hostFileBuffer, reorderedFileBuffer;
    BigBoy filesize = getFilesize(filename);
    
    BigBoy chunkSize = (argc == 1) ? 1024 : atoi(argv[1]);
    ChunkOrder *deviceReorderInfo, *hostReorderInfo;
    double *dScores, *hScores;
    uint8_t *dSyncBuffer;
    BigBoy n = (int) ceil(((double)filesize/(double)chunkSize));
    BigBoy nc2 = findNC2(n);
    unsigned int threadsPerBlock = 1024;
    unsigned int noOfBlocks = (int) ceil(((double)nc2/(double)threadsPerBlock));

    printf("File size = %llu; chunk size = %llu;    no of chunks = %llu;\n", filesize, chunkSize, n);
    printf("nc2 = %llu;       threadsPerBlock = %u; no of Blocks = %u; \n", nc2, threadsPerBlock, noOfBlocks);

    hostFileBuffer = (FileContent) malloc(filesize+1);
    reorderedFileBuffer = (FileContent) malloc(filesize+1);
    getFileContent(filename, hostFileBuffer);

    hostReorderInfo = (ChunkOrder *) malloc(sizeof(ChunkOrder)*n);
    hScores = (double *) malloc(sizeof(double)*n*n);

/*  
    checkCudaErrors(cudaMalloc((void **)&deviceFileBuffer, filesize));
    checkCudaErrors(cudaMalloc((void **)&dScores, sizeof(double)*nc2));
    checkCudaErrors(cudaMalloc((void **)&dSyncBuffer, sizeof(uint8_t)*10));

    checkCudaErrors(cudaMemcpy(deviceFileBuffer, hostFileBuffer, filesize, cudaMemcpyHostToDevice));

    printf("Launching CUDA kernal\n");

    sq<<<noOfBlocks, threadsPerBlock>>>(deviceFileBuffer, filesize, n, chunkSize, dScores, dSyncBuffer);
    checkCudaErrors(cudaMemcpy(hScores, dScores, sizeof(double)*nc2, cudaMemcpyDeviceToHost));
    printf("CUDA kernel execution over !!!\n");

    checkCudaErrors(cudaFree(deviceFileBuffer));
    checkCudaErrors(cudaFree(dScores));
    checkCudaErrors(cudaFree(dSyncBuffer));
*/

    hScores[0] = ggetScore(hostFileBuffer, 
                            chunkSize, 
                            hostFileBuffer + 1*chunkSize, 
                            min(chunkSize, filesize - chunkSize));
    //hScores[0] = score;

    for(int i = 0; i < nc2; i++) {
        printf("%f\n",hScores[i]);
    }

/*
    unsigned int hostOrderInfo[100]={};
    hostOrderInfo[0] = 0;
    lastFilled = 0;
    for(unsigned int i = 1; i < n; i++) {
        unsigned int start = (lastFilled      * n - (lastFilled       * (lastFilled + 1))/2);
        unsigned int end =   (lastFilled + 1) * n - ((lastFilled + 1) * (lastFilled + 2))/2;

        unsigned int j = 1, bestMatch;
        while(1) {
            bestMatch = lastFilled + getKthbestMatch(hScores + start, end - start, j);
            uint8_t  isFilled  = isAlreadyFilled(hostOrderInfo, i, bestMatch);
            if(!isFilled) break;
            j++;
        }
        hostOrderInfo[i] = bestMatch;
        lastFilled = bestMatch; 
    }
*/
    return 0;
}
