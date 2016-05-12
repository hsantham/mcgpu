
#include<stdio.h>
#include <sys/stat.h>
#include <assert.h>
#include <stdint.h>

#define MAX_INDEXES 10000
unsigned char *filename  = "reorder_info";
size_t getFilesize(unsigned char * filename) {
    struct stat st;
    if(stat((const char *)filename, &st) != 0) {
        assert(0);
        return 0;
    }
    return st.st_size;   
}

void getOrderFromFile(unsigned int *hScores, unsigned int *used) {
    FILE *file;
    size_t nread;

    char *fn = (char *) filename;
    file = fopen((const char *)fn, "r");
    assert(file != NULL);

    unsigned long long int filesize = getFilesize(fn);

    char *buffer = (char *) malloc(filesize+1);
    assert(buffer != NULL);
    nread = fread(buffer, 1, filesize, file);
    assert(nread == filesize);
    assert(ferror(file) == 0);

    int startIdx = 0, k = 0;
    for(int i=0; i<nread; i++) {
        if(buffer[i] == ' ') {
            buffer[i] = '\0';
            hScores[k++] = atoi(buffer+startIdx);
            startIdx = i + 1;           
        }

		assert(k<MAX_INDEXES);
    }
    hScores[k++] = atoi(buffer+startIdx);
	*used = k;
    fclose(file);
}

void getFileContent(unsigned char * filename, unsigned char * buffer) {
    FILE *file;
    size_t nread;

    file = fopen((const char *)filename, "r");
    assert(file != NULL);

    unsigned long long int filesize = getFilesize(filename);

    nread = fread(buffer, 1, filesize, file);
    assert(nread == filesize);
    assert(ferror(file) == 0);

    fclose(file);
}

unsigned int min(unsigned int a, unsigned int b) {
	return a<b ? a : b;
}

unsigned int getPosof(unsigned int *indexOrder, unsigned  int used, unsigned int index) {
for(int i=0; i<used; i++) {
    if(indexOrder[i] == index)
        return i;
}

assert(0);
return 0;
}

int main() {
	unsigned int indexOrder[MAX_INDEXES];
	unsigned int used;
	getOrderFromFile(indexOrder, &used);
	unsigned int chunksize = indexOrder[used-1];
	used--;
	
    unsigned char  *unziped   = "compress_with_mc";
	unsigned char *hostFileBuffer;
	unsigned long long int filesize = getFilesize(unziped);
	hostFileBuffer = (unsigned char *) malloc(filesize+1);
    getFileContent(unziped, hostFileBuffer);
	
	FILE *file;
    file = fopen((const char *)"decompressed_file", "w");
    assert(file != NULL);

    for(int i = 0; i < used; i++) {
        unsigned int jumbledPos = getPosof(indexOrder, used, i);
        unsigned int lastChunkPos = getPosof(indexOrder, used, used-1);
        unsigned int bytesToWrite = min(chunksize, filesize - i * chunksize);
        unsigned int offset = 0;
        if(lastChunkPos < jumbledPos) {
            offset =  chunksize - min(chunksize, filesize - (used-1) * chunksize);
        }
        unsigned int written = fwrite(hostFileBuffer + jumbledPos * chunksize - offset, 1, bytesToWrite, file);
        assert(written == bytesToWrite);
        assert(ferror(file) == 0);
    }

    fclose(file);
	return 0;
}
