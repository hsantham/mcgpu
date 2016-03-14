extern "C"
{
#include<stdio.h>
#include <sys/stat.h>
#include <assert.h>
#include <stdint.h>
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

struct llnode {
    void *data;
    struct llnode *next;
};

typedef struct llnode LLNode;

struct hnode {
    unsigned char *string;
    unsigned char source;
    unsigned char count;
};
typedef struct hnode HNode;

__device__ LLNode *getNewLLNode(void *data) {
    LLNode *head = (LLNode *) malloc(sizeof(LLNode));
    head->data = data;
    head->next = NULL;

    return head;
}

__device__ void append(LLNode *head, LLNode *nodeToInsert) {
  LLNode *node = head;

  while(node->next) node=node->next;

  node->next = nodeToInsert;
  nodeToInsert->next = NULL;
}

extern "C"
__device__ __host__ BigBoy findNC2(int n) {
return ((n * (n-1))/2);
}

// USER DEFINED TYPE DEFs
typedef struct chunkOrder ChunkOrder;

// Global variables
FileName filename;

extern "C"
__device__ void preprocess(FileContent   fileContent, int myIdx, int myLimit) {
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

__device__ uint32_t getHash(FileContent string, uint32_t length) {
    uint32_t hash = 0;
    for(uint32_t i = 0; i < length; i++) {
        hash = hash + string[i];
    }

    return hash % MAX_HASH_TABLE_ENTRIES;
}

__device__ HNode* getHashNode(FileContent string, 
                   uint8_t     source, 
                   uint8_t     count, 
                   HNode*      nodes, 
                   uint32_t*   used) {
    HNode* node = nodes + *used;
    node->string = string;
    node->source = source;
    node->count  = count;

    *(used)++;
    return node;
} 

__device__ void fillHashTable(LLNode **hashTable, uint32_t hash, LLNode *node) {
    LLNode *head = hashTable[hash];
    if(head == NULL) {
        hashTable[hash] = head;
    } else {
        while(head->next) head = head->next;
        head->next = node;
    }
}

__device__ LLNode* findMatchingNode(LLNode* head, FileContent content) {
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

extern "C"
__device__ double getScoreFromHashTable(LLNode **hashTable) {
    double euclidDist = 0;
    for(uint32_t i = 0; i < MAX_HASH_TABLE_ENTRIES; i++) {
        LLNode *node = hashTable[i];
        while(node) {
            HNode *hnode = (HNode *) node->data;            
            euclidDist += hnode->count * hnode->count;
            node = node->next;
        }
    }

    return sqrt((double) euclidDist);
}

extern "C"
__device__ double getScore(FileContent   content1, 
                           unsigned int  letterCount1, 
                           FileContent   content2, 
                           unsigned int  letterCount2) {
    LLNode *hashTable[MAX_HASH_TABLE_ENTRIES] = {};
    
    int i = 0, startIdx=-1;
    HNode nodes[MAX_NODES_IN_HASH_TABLE]; uint32_t used = 0;
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

extern "C"
__global__ void sq(FileContent   fileContent, 
                   BigBoy        fileSize, 
                   BigBoy        n,
                   BigBoy        chunkSize,
                   ChunkOrder    *reorderInfo)
{
    int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int myLimit = myIdx + chunkSize;

    int nc2 = findNC2(n);
    if(nc2 <= myIdx) return;

    preprocess(fileContent, myIdx, myLimit);
    __syncthreads();

    FileContent myContent = fileContent + myIdx;
    *myContent = '0' + myIdx/10;
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
unsigned int threadsPerBlock = 1024;
unsigned int noOfBlocks = (int) ceil(((double)nc2/(double)threadsPerBlock));

hostFileBuffer = (FileContent) malloc(filesize+1);
getFileContent(filename, hostFileBuffer);

checkCudaErrors(cudaMalloc((void **)&deviceFileBuffer, filesize));
checkCudaErrors(cudaMalloc((void **)&deviceReorderInfo, sizeof(ChunkOrder)*n));
hostReorderInfo = (ChunkOrder *) malloc(sizeof(ChunkOrder)*n);


checkCudaErrors(cudaMemcpy(deviceFileBuffer, hostFileBuffer, filesize, cudaMemcpyHostToDevice));

printf("Launching CUDA kernal for file size = %llu; chunk size = %llu; no of chunks = %llu;\n", filesize, chunkSize, n);
printf("                          threadsPerBlock = %u; noOfBlocks = %u; \n", threadsPerBlock, noOfBlocks);

//sq<<<1, 1>>>(deviceFileBuffer, filesize, n, chunkSize, deviceReorderInfo);
sq<<<noOfBlocks, threadsPerBlock>>>(deviceFileBuffer, filesize, n, chunkSize, deviceReorderInfo);

printf("CUDA kernel execution over !!!\n");

checkCudaErrors(cudaMemcpy(hostReorderInfo, deviceReorderInfo, sizeof(ChunkOrder)*n, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(hostFileBuffer, deviceFileBuffer, filesize, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaFree(deviceReorderInfo));
checkCudaErrors(cudaFree(deviceFileBuffer));
return 0;
}
