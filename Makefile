CC = g++
USER_LIBS = /libs/include
CUDA_INC = /usr/local/cuda/include
CUDA_LINK = /usr/local/cuda/lib64
USER_LINKS = /libs/lib
INC_DIR = $(USER_LIBS)
LINK_DIR = $(CUDA_LINK) $(USER_LINKS) 
INC_PARAMS = $(foreach d, $(INC_DIR), -I$d)
LINK_PARAMS = $(foreach l, $(LINK_DIR), -L$l)
CFLAGS = -lstdc++fs -fPIC -std=c++17 -Wno-deprecated -fopenmp $(LINK_PARAMS) $(INC_PARAMS) -lsqlite3 -lcuda -lcudart -lcublas -lfaiss -lopenblas -llapack

main : main.o
	$(CC) -g -o main main.o $(CFLAGS)

main.o : main.cpp
	$(CC) -g -c main.cpp $(CFLAGS) -o main.o

clean :
	rm -f *.o
