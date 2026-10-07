#pragma once
#include <vector>
#include <cmath>
#include <iostream> // for std::ostream

typedef unsigned long long uint64;

struct point {
    std::vector<double> v;
    int label;

    point(const std::vector<double>& coordinate, int label = -1);

    double& operator[](int idx);
    const double& operator[](int idx) const;

    int size() const;
    double distance(const point& other) const;
};

// Hash support for unordered containers
template<> struct std::hash<point> {
    std::size_t operator()(const point& p) const;
};

// Equality operator
bool operator==(const point &a, const point &b);

// Output to stream (for cout)
std::ostream& operator<<(std::ostream& os, const point& p);

// Utility: convert matrix to vector<point>
std::vector<point> convert_to_points(const std::vector<std::vector<double>>& B);
