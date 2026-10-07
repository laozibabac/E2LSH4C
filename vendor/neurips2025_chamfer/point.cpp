#include "point.h"

Point::Point() {}
Point::Point(const std::vector<double> & coords) : coords(coords) {}

std::ostream& operator<<(std::ostream& os, const Point & p) {
	os << "(";
	for (int i = 0; i < (int)p.coords.size(); ++i) {
		if (i) os << ", ";
		os << p.coords[i];
	}
	os << ")";
	return os;
}

int Point::dim() const { return coords.size(); }
double Point::coord(int i) const { return coords[i]; }
double Point::distance(const Point& p) const {
	double sum = 0;
	for (int i = 0; i < (int)coords.size(); ++i) {
		double d = coords[i] - p.coords[i];
		sum += d * d;
	}
	return std::sqrt(sum);
}
bool Point::operator<(const Point& p) const {
	for (int i = 0; i < (int)coords.size(); ++i)
		if (coords[i] != p.coords[i]) return coords < p.coords;
	return 0;
}
bool Point::operator != (const Point & p) const {
	return coords != p.coords;
}
