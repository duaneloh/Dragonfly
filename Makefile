.PHONY: mkdir
CC=gcc
CFLAGS = -Wno-unused-result -O3 -Wall -fopenmp
#CFLAGS = -O3 -Wall -fopenmp 
LIBS = -lm -lhdf5 -lgsl -lgslcblas -lfftw3 -fopenmp

src = $(wildcard src/*.c)
obj = $(patsubst src/%.c, bin/%.o, $(src))
utilssrc = $(wildcard utils/src/*.c)
utils = $(patsubst utils/src/%.c, utils/%, $(utilssrc))
directories = data bin

all: mkdir emc $(utils)

mkdir: $(directories)

$(directories):
	mkdir -p $(directories)

emc: $(obj)
	mpicc -o $@ $^ $(LIBS)

$(obj): bin/%.o: src/%.c src/emc.h
	mpicc -c $< -o $@ $(CFLAGS)

$(utils): utils/%: utils/src/%.c
	gcc -o $@ $< $(CFLAGS) $(LIBS)

clean:
	rm -f recon $(utils) $(obj)