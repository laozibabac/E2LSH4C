#include "BruteForceNN.h"
#include <limits>

BruteForceNN::BruteForceNN() {}

void BruteForceNN::insert(const point& p) {
    points.push_back(p);
}

double BruteForceNN::query(const point& p) const {
    if (points.empty()) throw std::runtime_error("Dataset is empty and cannot be processed.");

    point best = points[0];
    double best_dist = p.distance(best);

    for (const auto& candidate : points) {
        double d = p.distance(candidate);
        if (d < best_dist) {
            best = candidate;
            best_dist = d;
        }
    }
    return best_dist;
}
