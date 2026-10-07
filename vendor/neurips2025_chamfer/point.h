#pragma once
#include <cmath>
#include <vector>
#include <fstream>
#include "random.h"

class Point {
public:
	std::vector<double> coords;
	Point();
	Point(const std::vector<double> & coords);
	int dim() const;
	double coord(int i) const;
	double distance(const Point& p) const;
	bool operator<(const Point& p) const;
	bool operator != (const Point & p) const;

	friend std::ostream& operator<<(std::ostream& os, const Point & p);
};

