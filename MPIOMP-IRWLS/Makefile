CC=mpicc
OSX=0

OPTFLAGS = -fPIC -O3 -fopenmp

BINFOLDER := bin
BUILDFOLDER := build
SRCFOLDER := src

SRCEXT := c
SOURCES := $(shell find $(SRCFOLDER) -type f -name "*.$(SRCEXT)")
OBJECTS := $(patsubst $(SRCFOLDER)/%,$(BUILDFOLDER)/%,$(SOURCES:.$(SRCEXT)=.o))

INCLUDE  = -Iinclude

LIBS = -lm -llapack -lf77blas -lcblas -latlas -lgfortran -fopenmp

CCOPTION = 

ifeq ($(OSX),1)
    LIBRARYPATH = -L/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/
    LIBS = -lLAPACK -lBLAS -fopenmp
    CCOPTION = -DOSX
endif

ifdef VECLIBDIR
    LIBRARYPATH = -L$(VECLIBDIR)/
endif

ifdef ATLASDIR
    LIBRARYPATH = -L$(ATLASDIR)/lib/
endif

ifeq ($(MKL),1)
    #CC=/opt/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/bin/mpicc
    #INCLUDEPATH = -I/opt/intel/mkl/include/
    #LIBRARYPATH = -L/opt/intel/mkl/lib/intel64/ 
    LIBS = -lm -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blas95_lp64 -lmkl_lapack95_lp64 -fopenmp
    CCOPTION = -DMKL
endif

COMMONOBJ := $(BUILDFOLDER)/ParallelAlgorithms.o $(BUILDFOLDER)/IOStructures.o $(BUILDFOLDER)/kernels.o $(BUILDFOLDER)/LIBIRWLS-predict.o $(BUILDFOLDER)/PSIRWLS-train.o

all: PSIRWLS-train LIBIRWLS-predict

PSIRWLS-train: $(BUILDFOLDER)/ExecPSIRWLS-train.o $(COMMONOBJ)
	@echo " Linking PSIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)

LIBIRWLS-predict: $(BUILDFOLDER)/ExecLIBIRWLS-predict.o $(COMMONOBJ)
	@echo " Linking PSIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)

$(BUILDFOLDER)/%.o: $(SRCFOLDER)/%.$(SRCEXT)
	@echo " mkdir -p $(BUILDFOLDER)"; mkdir -p $(BUILDFOLDER)
	@echo " $(CC) $(CCOPTION) $(OPTFLAGS) $(CFLAGS) $(INCLUDE) $(INCLUDEPATH) $(LIBRARYPATH) -c -o $@ $<"; $(CC) $(CCOPTION) $(OPTFLAGS) $(CFLAGS) $(INCLUDE) $(INCLUDEPATH) $(LIBRARYPATH) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " rm -rf $(BINFOLDER) $(BUILDFOLDER)"; rm -rf $(BINFOLDER) $(BUILDFOLDER)

.PHONY: clean
