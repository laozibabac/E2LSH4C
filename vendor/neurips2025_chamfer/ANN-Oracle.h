#pragma once
#include <set>
#include <map>
#include <iostream>
#include "ANN.h"
#include "point.h"
#include "misc.h"

using std::cout;
using std::endl;

class ANNOracle {
	ANN *ann;
	int cnt, _size, limit;
	std::map<Point, ANN::pointer> *ptr_map;
	std::set<Point> *point_set;

 public:
	ANNOracle();
	ANNOracle(const std::vector<Point>& points);

	ANNOracle(const ANNOracle &) = delete;
	ANNOracle& operator=(const ANNOracle &) = delete;

	Point findNN(const Point &x);
	double distance(const Point& x);
	void insert(const Point& x);
	void erase(const Point& x);
	bool contains(const Point& x);
	int size() const;
};