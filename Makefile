CC = g++
USER_LIBS = /libs/include
CUDA_INC = /usr/local/cuda/include
INC_DIR = $(USER_LIBS) $(CUDA_INC)
INC_PARAMS = $(foreach d, $(INC_DIR), -I$d)
CFLAGS = -lstdc++fs -fPIC -std=c++17 -Wno-deprecated -fopenmp $(INC_PARAMS) -lfaiss -lcuda -lopenblas -llapack

main : main.o
	$(CC) -g -o main main.o $(CFLAGS)

main.o : main.cpp
	$(CC) -g -c main.cpp $(CFLAGS) -o main.o

clean :
	rm -f *.o
