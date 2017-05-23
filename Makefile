.PHONY: mkdir env

MPICC = mpicc
CC = gcc

CFLAGS = $(shell gsl-config --cflags) -Wno-unused-result -O3 -Wall -fopenmp
LIBS = $(shell gsl-config --libs) -fopenmp

OMPI_CC = $(CC)
EMC_HEADER = $(wildcard src/*.h)
EMC_SRC = $(wildcard src/*.c)
EMC_OBJ = $(patsubst src/%.c, bin/%.o, $(EMC_SRC))
UTILS_SRC = $(wildcard utils/src/*.c)
UTILS = $(patsubst utils/src/%.c, utils/%, $(UTILS_SRC))
UTILS := $(filter-out utils/compare, $(UTILS))
UTILS := $(filter-out utils/make_quaternion, $(UTILS))
UTILS := $(filter-out utils/make_data, $(UTILS))
DIRECTORIES = data bin images data/output data/orientations data/mutualInfo data/weights data/scale data/likelihood

all: mkdir emc $(UTILS) utils/compare utils/make_quaternion utils/make_data

mkdir: $(DIRECTORIES)
$(DIRECTORIES):
	mkdir -p $(DIRECTORIES)

emc: $(EMC_OBJ)
ifeq ($(OMPI_CC), gcc)
	$(MPICC) -o $@ $^ $(LIBS)
else
	`export OMPI_CC=$(OMPI_CC); $(MPICC) -o $@ $^ $(LIBS)`
endif

$(EMC_OBJ): bin/%.o: src/%.c
ifeq ($(OMPI_CC), gcc)
	$(MPICC) -c $< -o $@ $(CFLAGS)
else
	`export OMPI_CC=$(OMPI_CC); $(MPICC) -c $< -o $@ $(CFLAGS)`
endif

bin/recon_emc.o bin/setup_emc.o bin/max_emc.o: $(EMC_HEADER)
bin/detector.o: src/detector.h
bin/dataset.o: src/dataset.h src/detector.h
bin/interp.o: src/interp.h src/detector.h
bin/quat.o: src/quat.h
bin/iterate.o: src/iterate.h src/dataset.h src/detector.h

$(UTILS): utils/%: utils/src/%.c
	$(CC) -o $@ $< $(CFLAGS) $(LIBS)

utils/compare: utils/src/compare.c bin/quat.o bin/interp.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

utils/make_quaternion: utils/src/make_quaternion.c bin/quat.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

utils/make_data: utils/src/make_data.c bin/detector.o bin/interp.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -f emc $(UTILS) $(EMC_OBJ)
