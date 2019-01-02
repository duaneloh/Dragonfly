.PHONY: mkdir clean unit_test

MPICC = mpicc
CC = gcc

OMPI_CC = $(CC)
ifneq (,$(findstring icc, $(CC)))
	OMP_FLAG = -qopenmp
else
	OMP_FLAG = -fopenmp
endif
CFLAGS = $(shell gsl-config --cflags) -O3 -Wall $(OMP_FLAG)
LIBS = $(shell gsl-config --libs) $(OMP_FLAG) -Wl,-rpath,$(shell gsl-config --prefix)/lib

# Derive source files and targets
# ============================================================
EMC_HEADER = $(wildcard src/*.h)
EMC_SRC = $(wildcard src/*.c)
EMC_OBJ = $(patsubst src/%.c, bin/%.o, $(EMC_SRC))
UTILS_SRC = $(wildcard utils/src/*.c)
UTILS = $(patsubst utils/src/%.c, utils/%, $(UTILS_SRC))
DIRECTORIES = data bin images

# Create directories and compile and link C code
# ============================================================
all: mkdir emc $(UTILS)

# Make directories which are not shipped with the repository
# ============================================================
mkdir: $(DIRECTORIES)
$(DIRECTORIES):
	mkdir -p $(DIRECTORIES)

# Link emc executable from various objects
emc: $(EMC_OBJ)
ifeq ($(OMPI_CC), gcc)
	$(MPICC) -o $@ $^ $(LIBS)
else
	`export OMPI_CC=$(OMPI_CC); $(MPICC) -o $@ $^ $(LIBS)`
endif

# Compile emc objects
# ============================================================
bin/recon_emc.o bin/setup_emc.o bin/max_emc.o: $(EMC_HEADER)
bin/detector.o: src/detector.h
bin/dataset.o: src/dataset.h src/detector.h
bin/interp.o: src/interp.h src/detector.h
bin/quat.o: src/quat.h
bin/params.o: src/params.h
bin/iterate.o: src/iterate.h src/dataset.h src/detector.h src/params.h

$(EMC_OBJ): bin/%.o: src/%.c
ifeq ($(OMPI_CC), gcc)
	$(MPICC) -c $< -o $@ $(CFLAGS)
else
	`export OMPI_CC=$(OMPI_CC); $(MPICC) -c $< -o $@ $(CFLAGS)`
endif

# Compile and link C utilities
# ============================================================
utils/compare: bin/quat.o bin/interp.o
utils/make_quaternion: bin/quat.o
utils/make_data: bin/detector.o bin/interp.o
utils/merge: bin/detector.o bin/dataset.o bin/interp.o bin/iterate.o
utils/fiberize: bin/detector.o bin/interp.o
utils/cosmic: bin/detector.o bin/dataset.o

$(UTILS): utils/%: utils/src/%.c
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

# Cythonize C code and run unit tests
# ============================================================
test: all unit_test

unit_test:
	./utils/run_tests.sh

# Remove compiled files
# ============================================================
clean:
	rm -f emc $(UTILS) $(EMC_OBJ)
	rm -rf pyemc/build pyemc/*.so
