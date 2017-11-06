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
DIRECTORIES = data bin images

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

bin/recon_emc.o bin/setup_emc.o bin/max_emc.o: $(EMC_HEADER)
bin/detector.o: src/detector.h
bin/dataset.o: src/dataset.h src/detector.h
bin/interp.o: src/interp.h src/detector.h
bin/quat.o: src/quat.h
bin/iterate.o: src/iterate.h src/dataset.h src/detector.h

$(EMC_OBJ): bin/%.o: src/%.c
ifeq ($(OMPI_CC), gcc)
	$(MPICC) -c $< -o $@ $(CFLAGS)
else
	`export OMPI_CC=$(OMPI_CC); $(MPICC) -c $< -o $@ $(CFLAGS)`
endif

utils/compare: bin/quat.o bin/interp.o
utils/make_quaternion: bin/quat.o
utils/make_data: bin/detector.o bin/interp.o
utils/merge: bin/detector.o bin/dataset.o bin/interp.o
utils/fiberize: bin/detector.o bin/interp.o

$(UTILS): utils/%: utils/src/%.c
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -f emc $(UTILS) $(EMC_OBJ)
