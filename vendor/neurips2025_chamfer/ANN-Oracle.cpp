#include "ANN-Oracle.h"
#include <cassert>
#include <iostream>
#include <cmath>

ANNOracle::ANNOracle() {
	cnt = 0; _size = 0;
	ann = nullptr; ptr_map = nullptr;
	point_set = new std::set<Point>();
	limit = int((pow(dataset_window_size, dataset_rho) + 2) * 20);
}

ANNOracle::ANNOracle(const std::vector<Point>& points) {
	cnt = 0; _size = 0;
	ann = nullptr; ptr_map = nullptr;
	point_set = new std::set<Point>();
	limit = int((pow(dataset_window_size, dataset_rho) + 2) * 20);
	for (const Point &p : points) insert(p);
}

Point ANNOracle::findNN(const Point &x) {
	if (!ann) {
		Point best = *point_set->begin();
		for (const auto& p : *point_set) {
			if (p.distance(x) < best.distance(x)) best = p;
		}
		return best;
	} else {
		return Point(ann->nearest_neighbor(point(x.coords, -1)).v);
	}
}

double ANNOracle::distance(const Point& x) {
	if (_size == 0) return INFINITY;
	Point p = findNN(x);
	return p.distance(x);
}

void ANNOracle::insert(const Point& x) {
	_size++;
	if (_size == limit && !ann) {
		ann = new ANN(dataset_window_size, dataset_dmin, dataset_dmax, dataset_rho);
		ptr_map = new std::map<Point, ANN::pointer>();
		cnt = 0;
		for (const Point &p : *point_set) {
			auto it = ann->insert(point(p.coords, cnt));
			(*ptr_map)[p] = it;
			cnt++;
		}
		delete point_set;
		point_set = nullptr;
	}

	if (!ann) {
		point_set->insert(x);
	} else {
		ANN::pointer ptr = ann->insert(point(x.coords, cnt));
		if (ptr_map->count(x) == 0){
			(*ptr_map)[x] = ptr;
			cnt++;
		}
	}
}

void ANNOracle::erase(const Point& x) {
	_size--;
	if (!ann) {
		point_set->erase(x);
	} else {
		auto it = ptr_map->find(x);
	
		assert(it != ptr_map->end());
		ann->remove(it->second);
		ptr_map->erase(it);
	}
}

bool ANNOracle::contains(const Point& x) {
	if (!ann) return point_set->count(x) > 0;
	else return ptr_map->count(x) > 0;
}

int ANNOracle::size() const { return _size; }
