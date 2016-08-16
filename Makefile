.PHONY: mkdir env

MPICC = mpicc
CC = gcc

CFLAGS = -Wno-unused-result -O3 -Wall -fopenmp
LIBS = -lm -lgsl -lgslcblas -fopenmp

EMC_SRC = $(wildcard src/*.c)
EMC_OBJ = $(patsubst src/%.c, bin/%.o, $(src))
UTILS_SRC = $(wildcard utils/src/*.c)
UTILS = $(patsubst utils/src/%.c, utils/%, $(utilssrc))
DIRECTORIES = data bin images data/output data/orientations data/mutualInfo data/weights data/scale

all: env mkdir emc $(UTILS)

mkdir: $(DIRECTORIES)

env:
	export OMPI_CC=$(CC)

$(DIRECTORIES):
	mkdir -p $(DIRECTORIES)

emc: $(EMC_OBJ)
	$(MPICC) -o $@ $^ $(LIBS)

$(EMC_OBJ): bin/%.o: src/%.c src/emc.h
	$(MPICC) -c $< -o $@ $(CFLAGS)

$(UTILS): utils/%: utils/src/%.c
	$(CC) -o $@ $< $(CFLAGS) $(LIBS)

utils/compare: src/quat.c

clean:
	rm -f emc $(UTILS) $(EMC_OBJ)
