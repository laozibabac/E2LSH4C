#pragma once
#include <vector>
#include <random>
#include "dataset_config.h"
#include "datatype.h"

struct DSU {
	int *fa;
	int size;

	DSU(int size);
	int root(int x);
	bool join(int x, int y);
};

bool is_prime(int x);
double dist(const point &x, const point &y);
double dot(const point &x, const point &y);

extern std::mt19937 global_rng;
point rand_unit_vector();
point bad_point_value();


double benchmark(const std::vector<point> &points, const std::vector<point> &facilities, double opening_cost);