CC = g++
CFLAGS = -std=c++17

main : main.o
	$(CC) -g -o main main.o $(CFLAGS)

main.o : main.cpp
	$(CC) -g -c main.cpp $(CFLAGS) -o main.o


	