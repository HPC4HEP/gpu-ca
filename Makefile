IDIR =./include
SRCDIR=./src
CC=g++
NVCC=nvcc
CFLAGS=-I$(IDIR) -std=c++11
CUDAFLAGS=
ODIR=./obj
LDIR =./lib

OUTPUTDIR=./bin
#LIBS=-lm

_DEPS = Cell.h CUDAQueue.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))


_OBJ = GPUCA.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
$(ODIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(NVCC) –default-stream per-thread -c -o $@ $< $(CUDAFLAGS) $(CFLAGS)


GPUCA: $(OBJ)
	$(NVCC) –default-stream per-thread -o $(OUTPUTDIR)/$@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 