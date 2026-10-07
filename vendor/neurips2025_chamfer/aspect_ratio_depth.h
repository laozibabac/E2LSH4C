#ifndef ASPECT_RATIO_DEPTH_H
#define ASPECT_RATIO_DEPTH_H

#include <vector>
#include <cmath>

/**
 * @param p1 
 * @param p2 
 * @return 
 */
double euclidean_distance(const std::vector<double>& p1, const std::vector<double>& p2);

/**
 * @param P 
 * @return 
 */
double compute_2approx_diameter(const std::vector<std::vector<double> >& P);

/**
 * @param A 
 * @param B 
 * @return 
 */ 
double calculate_aspect_ratio(const std::vector<std::vector<double> >& A,
                            const std::vector<std::vector<double> >& B, double max_dist);

/**
 * @param aspect_ratio 
 * @param d
 * @return 
 */
int calculate_quadtree_depth(double aspect_ratio);

#endif // ASPECT_RATIO_DEPTH_H
