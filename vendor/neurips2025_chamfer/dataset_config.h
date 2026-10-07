#pragma once
#include "datatype.h"
#include <string>

// extern const int DIM;
extern const double dataset_rho;
extern const double dataset_dmin;
extern const double dataset_dmax;
extern const int dataset_window_size;
// Must exceed max A index (QuadTree packs B ids as local_index + B_ID_shift).
// 10M+1 covers GloVe ~1.18M and DEEP1B-scale A; SIFT/GIST 1_000_001 still fit.
inline constexpr int B_ID_shift = 10'000'001;
// DIM is compile-time (bitset<DIM>). Override via -DCHAMFER_DIM=128 for SIFT or
// -DCHAMFER_DIM=960 for GIST without changing Fashion (DIM=784) default.
// See Makefile compare_perf_sift / compare_perf_gist.
#ifndef CHAMFER_DIM
#define CHAMFER_DIM 784
#endif
inline constexpr std::size_t DIM = CHAMFER_DIM;
// Historical toggles (prefer CHAMFER_DIM / make DIM=128):
// inline constexpr std::size_t DIM = 300;
// inline constexpr std::size_t DIM = 128;
// inline constexpr std::size_t DIM = 2;
