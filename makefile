CC = g++
CLibs = `pkg-config --cflags --libs opencv`
RM = rm -f *.o
Target = main

all: main

filter: filter.cpp 
	$(CC) -c filter.cpp $(CLibs)

main: filter.o
	$(CC) -o $(Target) $(Target).cpp filter.o $(CLibs)
	$(RM)

clean:
	$(RM)
