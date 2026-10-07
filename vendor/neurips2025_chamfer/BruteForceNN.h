#pragma once
#include "datatype.h"
#include <vector>

class BruteForceNN {
public:
    BruteForceNN();

    void insert(const point& p);
    double query(const point& p) const;

private:
    std::vector<point> points;
};
