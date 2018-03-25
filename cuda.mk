export LANG=C

###########################################################
# COMPILES
###########################################################
ICC  = icc -DCOMP='"Intel C++ Compiler"' 
GCC  = g++ -DCOMP='"GNU C++ Compiler"' 
PCC  = pathCC -DCOMP='"Pathscale C++ Compiler"' 
IBM  = bgxlc++ -DCOMP='"IBM XLC++ Compiler"' 
MPICC= mpicxx -DMPICF 
NVCC:= nvcc -ccbin $(GCC)

###########################################################
# FLAGS
###########################################################
NVFLAGS       := -Xptxas="-v"
CCFLAGS       := -O3 -DDATA_VECTOR_SZ=3 -fPIC -g
EXTRA_CCFLAGS ?= #-std=c++11
EXTRA_NVFLAGS ?= -I./src/hdr -I./src/fld -std=c++11

GENCODE_SM20  := # -gencode arch=compute_20,code=sm_21 --maxrregcount 32
GENCODE_SM30  := #-gencode arch=compute_30,code=sm_30
GENCODE_SM35  := #-gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50  := -gencode arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM61  := #-gencode arch=compute_61,code=\"sm_61,compute_61\"
ALL_NVFLAGS   := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM61) $(NVFLAGS) $(EXTRA_NVFLAGS) 
ALL_CCFLAGS   := --compiler-options="$(CCFLAGS) $(EXTRA_CCFLAGS)" 
ALL_FLAGS     := $(ALL_CCFLAGS) $(ALL_NVFLAGS)

###########################################################
# LIBS
###########################################################
 GCCLIBS = -lstdc++ -lgomp -lpthread -lpng 
CUDALIBS = -L/usr/local/cuda/lib64/ -lcuda -lcudart -lglut -lGL -lcufft
ALL_LIBS = $(GCCLIBS) $(CUDALIBS)

###########################################################
# INCLUDES
###########################################################
SOURCE_HEADERS:=		\
       	cuda_math.h	\
       	tablet.hpp 	\
       	heap.hpp		\
	stencil.cuh		\
	FCC.h
ALL_HEADERS = $(SOURCE_HEADERS) 

###########################################################
# OBJECTS
###########################################################
OBJECTS := tablet_interface.o
###########################################################
# BUILD
###########################################################
TARGET = model 

all: libtablet.so tablet # build
libtablet.so:  tablet.o $(OBJECTS)
	$(EXEC) $(NVCC) $(ALL_FLAGS) $(ALL_LIBS) $(OBJECTS) --shared $< -o $@
	 
tablet:  tablet.o
	$(EXEC) $(NVCC) $(ALL_FLAGS) $(ALL_LIBS) $< -o $@
	 
build:  $(OBJECTS) rot.o
	$(EXEC) $(NVCC) $(ALL_FLAGS) $(ALL_LIBS) $(OBJECTS) rot.o -o $(TARGET)

tablet.o: tablet.cu $(ALL_HEADERS)
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(ALL_NVFLAGS) -o $@ -dc $<

rot.o: rot.cu $(ALL_HEADERS)
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(ALL_NVFLAGS) -o $@ -dc $<

$(OBJECTS): %.o: %.cu $(ALL_HEADERS)
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(ALL_NVFLAGS) -o $@ -dc $<

run: build
	./$(TARGET)

crun: build
	clear;
	./$(TARGET)

prof: build
	nvprof -m inst_fp_32 ./$(TARGET)

clean: 
	$(EXEC) rm -f $(OBJECTS) 
	$(EXEC) rm -f rot.o tablet.o
	$(EXEC) rm -f $(TARGET)
	rm -f libtablet.so
tidy:	
	@rm -f *~
