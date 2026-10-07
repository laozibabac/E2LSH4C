#include "aspect_ratio_depth.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

using namespace std;

double euclidean_distance(const vector<double>& p1, const vector<double>& p2) {
    double sum = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double compute_2approx_diameter(const vector<vector<double> >& P) {
    if (P.empty()) {
        return 0.0;
    }
    
    const vector<double>& u = P[0];
    double max_dist = 0.0;
    
    for (const auto& point : P) {
        double dist = euclidean_distance(u, point);
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    
    return max_dist * 2; 
}

double calculate_aspect_ratio(const vector<vector<double> >& A, const vector<vector<double> >& B, double max_dist) {
    double min_dist = 1e-4;
    cout << max_dist << " " << min_dist << endl;
    return max_dist / min_dist;
}

int calculate_quadtree_depth(double aspect_ratio) {
    return ceil(log2(aspect_ratio));
}
