.PHONY: mkdir env

MPICC = mpicc
CC = gcc

CFLAGS = $(shell gsl-config --cflags) -Wno-unused-result -O3 -Wall -fopenmp
LIBS = $(shell gsl-config --libs) -fopenmp

OMPI_CC = $(CC)
EMC_SRC = $(wildcard src/*.c)
EMC_OBJ = $(patsubst src/%.c, bin/%.o, $(EMC_SRC))
UTILS_SRC = $(wildcard utils/src/*.c)
UTILS = $(patsubst utils/src/%.c, utils/%, $(UTILS_SRC))
DIRECTORIES = data bin images data/output data/orientations data/mutualInfo data/weights data/scale

all: mkdir emc $(UTILS)

mkdir: $(DIRECTORIES)

$(DIRECTORIES):
	mkdir -p $(DIRECTORIES)

emc: $(EMC_OBJ)
ifeq ($(OMPI_CC), gcc)
	$(MPICC) -o $@ $^ $(LIBS)
else
	`export OMPI_CC=$(OMPI_CC); $(MPICC) -o $@ $^ $(LIBS)`
endif

$(EMC_OBJ): bin/%.o: src/%.c src/emc.h
ifeq ($(OMPI_CC), gcc)
	$(MPICC) -c $< -o $@ $(CFLAGS)
else
	`export OMPI_CC=$(OMPI_CC); $(MPICC) -c $< -o $@ $(CFLAGS)`
endif

$(UTILS): utils/%: utils/src/%.c
	$(CC) -o $@ $< $(CFLAGS) $(LIBS)

utils/compare: src/quat.c

clean:
	rm -f emc $(UTILS) $(EMC_OBJ)
