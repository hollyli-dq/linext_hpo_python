CXX ?= g++
USE_CUDA ?= no
CUFLAGS_GENCODE ?= -gencode=arch=compute_75,code=sm_75

LDFLAGS := -pthread
CFLAGS_COMMON := -std=c++14 -Wall -march=native -MMD
ifeq ($(USE_CUDA),yes)
	LDFLAGS := $(LDFLAGS) -lcuda -lcudart
	CFLAGS_COMMON := $(CFLAGS_COMMON) -DLINEXT_USE_CUDA=1
endif
CFLAGS_RELEASE := $(CFLAGS_COMMON) -O3 -DNDEBUG
CFLAGS_DEBUG := $(CFLAGS_COMMON) -Og -g

CUFLAGS_COMMON := $(CUFLAGS_GENCODE) -DLINEXT_USE_CUDA=1
CUFLAGS_RELEASE := $(CUFLAGS_COMMON) -DNDEBUG
CUFLAGS_DEBUG := $(CUFLAGS_COMMON) -g -G

CPP_SRCS := $(shell find src -name '*.cpp')
CPP_OBJS_RELEASE := $(CPP_SRCS:%.cpp=%.cpp.release.o)
CPP_OBJS_DEBUG := $(CPP_SRCS:%.cpp=%.cpp.debug.o)
CPP_OBJS := $(CPP_OBJS_RELEASE) $(CPP_OBJS_DEBUG)
CPP_DEPS_RELEASE := $(CPP_SRCS:%.cpp=%.cpp.release.d)
CPP_DEPS_DEBUG := $(CPP_SRCS:%.cpp=%.cpp.debug.d)
CPP_DEPS := $(CPP_DEPS_RELEASE) $(CPP_DEPS_DEBUG)
CU_SRCS := $(shell find src -name '*.cu')
CU_OBJS_RELEASE := $(CU_SRCS:%.cu=%.cu.release.o)
CU_OBJS_DEBUG := $(CU_SRCS:%.cu=%.cu.debug.o)
CU_OBJS := $(CU_OBJS_RELEASE) $(CU_OBJS_DEBUG)

ifeq ($(USE_CUDA),yes)
	OBJS_RELEASE := $(CPP_OBJS_RELEASE) $(CU_OBJS_RELEASE)
	OBJS_DEBUG := $(CPP_OBJS_DEBUG) $(CU_OBJS_DEBUG)
else
	OBJS_RELEASE := $(CPP_OBJS_RELEASE)
	OBJS_DEBUG := $(CPP_OBJS_DEBUG)
endif

.PHONY: all clean instances

all: linext linext.debug instances

instances:
	$(MAKE) -C instances

linext: $(OBJS_RELEASE)
	$(CXX) $(CFLAGS_RELEASE) $(OBJS_RELEASE) -o linext $(LDFLAGS)

linext.debug: $(OBJS_DEBUG)
	$(CXX) $(CFLAGS_DEBUG) $(OBJS_DEBUG) -o linext.debug $(LDFLAGS)

%.cpp.release.o: %.cpp
	$(CXX) $(CFLAGS_RELEASE) -c $< -o $@

%.cpp.debug.o: %.cpp
	$(CXX) $(CFLAGS_DEBUG) -c $< -o $@

%.cu.release.o: %.cu %.hpp src/common.hpp src/pcg32.hpp
	nvcc $(CUFLAGS_RELEASE) -c $< -o $@

%.cu.debug.o: %.cu %.hpp src/common.hpp src/pcg32.hpp
	nvcc $(CUFLAGS_DEBUG) -c $< -o $@

src/relaxtpa_gpu.cu.release.o: src/relaxtpa_gpu_kernel.cuh

src/relaxtpa_gpu.cu.debug.o: src/relaxtpa_gpu_kernel.cuh

clean:
	rm -f linext linext.debug $(CPP_OBJS_RELEASE) $(CU_OBJS_RELEASE) $(CPP_OBJS_DEBUG) $(CU_OBJS_DEBUG) $(CPP_DEPS)

-include $(CPP_DEPS)
