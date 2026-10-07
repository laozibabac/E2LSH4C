# Thin GNU Make wrapper around CMake + Ninja.
# Do not use GnuWin32 make 3.81. On Windows:
#   cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
#   cmake --build build --target compare_perf compare_perf_sift compare_perf_gist compare_perf_d300 exact_cache_f32

BUILD_DIR ?= build
GENERATOR ?= Ninja
CMAKE ?= cmake

FAST_MATH ?= 0
EARLY_EXIT ?= 0

CMAKE_FLAGS =
ifeq ($(FAST_MATH),1)
  CMAKE_FLAGS += -DE2LSH_FAST_MATH=ON
endif
ifeq ($(EARLY_EXIT),1)
  CMAKE_FLAGS += -DE2LSH_EARLY_EXIT=ON
endif

.PHONY: all configure clean compare_perf_sift compare_perf_gist compare_perf_d300 exact_cache_f32

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -G $(GENERATOR) $(CMAKE_FLAGS)

all: configure
	$(CMAKE) --build $(BUILD_DIR) --target compare_perf compare_perf_sift compare_perf_gist compare_perf_d300 exact_cache_f32

compare_perf_sift: configure
	$(CMAKE) --build $(BUILD_DIR) --target compare_perf_sift

compare_perf_gist: configure
	$(CMAKE) --build $(BUILD_DIR) --target compare_perf_gist

compare_perf_d300: configure
	$(CMAKE) --build $(BUILD_DIR) --target compare_perf_d300

exact_cache_f32: configure
	$(CMAKE) --build $(BUILD_DIR) --target exact_cache_f32

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean 2>/dev/null || true
	rm -rf $(BUILD_DIR)
