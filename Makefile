.PHONY: mkdir
CFLAGS = -Wno-unused-result -O3 -Wall -fopenmp
LIBS = -lm -lgsl -lgslcblas -fopenmp

src = $(wildcard src/*.c)
obj = $(patsubst src/%.c, bin/%.o, $(src))
utilssrc = $(wildcard utils/src/*.c)
utils = $(patsubst utils/src/%.c, utils/%, $(utilssrc))
directories = data bin images data/output data/orientations data/mutualInfo data/weights data/scale

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

utils/compare: src/quat.c

clean:
	rm -f recon $(utils) $(obj)
