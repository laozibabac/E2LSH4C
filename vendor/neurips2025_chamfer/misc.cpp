#include "misc.h"
#include <cmath>
#include <algorithm>

std::mt19937 global_rng (4121);

DSU::DSU(int n) : size(n) {
	fa = new int[n];
	for (int i = 0; i < n; ++i) fa[i] = i;
}

int DSU::root(int x) {
	if (fa[x] == x) return x;
	else {
		fa[x] = root(fa[x]);
		return fa[x];
	}
}

bool DSU::join(int x, int y) {
	x = root(x);
	y = root(y);
	if (x != y) {
		fa[x] = y;
		return true;
	} else {
		return false;
	}
}

double dist(const point &x, const point &y) {
	double sum = 0;
	for (int i = 0; i < DIM; ++i) {
		sum += (x[i] - y[i]) * (x[i] - y[i]);
	}
	return sqrt(sum);
}

double dot(const point &x, const point &y) {
	double sum = 0;
	for (int i = 0; i < DIM; ++i) {
		sum += x[i] * y[i];
	}
	return sum;
}

point rand_unit_vector() {
	std::normal_distribution<> gaussian(0, 1);
	std::vector<double> v(DIM);
	double length = 0;
	for (int i = 0; i < DIM; ++i) {
		v[i] = gaussian(global_rng);
		length += v[i] * v[i];
	}
	length = sqrt(length);
	for (int i = 0; i < DIM; ++i)
		v[i] /= length;
	return point(v, -1);
}

point bad_point_value() {
	std::vector<double> v(DIM);
	return point(v, -1);
}

double benchmark(const std::vector<point> &points, const std::vector<point> &facilities, double opening_cost) {
	double cost = facilities.size() * opening_cost;
	for (const point &p : points) {
		double min_dist = 1e9;
		for (const point &f : facilities) {
			min_dist = std::min(min_dist, dist(p, f));
		}
		cost += min_dist;
	}
	return cost;
}

bool is_prime(int x) {
	for (int i = 2; i * i <= x; ++i) {
		if (x % i == 0) {
			return false;
		}
	}
	return true;
}