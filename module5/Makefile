# make          build assignment.exe
# make clean    remove it
#
# CUDA_ARCH is empty by default, so nvcc embeds PTX and the driver JIT-compiles
# for whatever GPU is present. Override to skip the JIT, e.g.
#   make CUDA_ARCH=-arch=sm_75     (Tesla T4, the usual Colab GPU)

NVCC      ?= nvcc
TARGET    := assignment.exe
SOURCES   := assignment.cu
CUDA_ARCH ?=
NVCCFLAGS := -O2 -lineinfo $(CUDA_ARCH)

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
