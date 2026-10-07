#include "misc.h"
#include <vector>
#include <cstdio>
#include <string>

// const int DIM = 300;
// A_size = 1880
// B_size = 1178
const double dataset_dmin = 1e-4;
const double dataset_dmax = 17.8444;
const double dataset_rho = 0.2;
const int dataset_window_size = 76; // B in window

// int window_size = (A.size() + B.size()) / 20 = 152;
// int query_interval = (A.size() + B.size()) / 50 = 61;
// sample_times_each_window = 30
// max_A_size_in_all_window = 76
// max_B_size_in_all_window = 76