#pragma once
// Exact nearest neighbor d_a = min_b ||a - b||_2.
// Both algorithms use exact NN, so the comparison isolates sampling.
// Scanning B is O(|B| d).
// E2LSH queries use ChamferEstimator::exactNN on the contiguous array B_flat_.
// This file is for QuadTree, Uniform, and harnesses that own a copy of B.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace exact_nn {

inline double squared_l2(const double* a, const double* b, int dim) {
  double s = 0.0;
  for (int i = 0; i < dim; ++i) {
    double d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

inline double l2(const std::vector<double>& a, const std::vector<double>& b) {
  return std::sqrt(squared_l2(a.data(), b.data(), static_cast<int>(a.size())));
}

// Min ℓ2 from query q to each point in B given as raw coordinate pointers.
inline double min_l2(const double* q, int dim, const std::vector<const double*>& B) {
  if (B.empty()) return std::numeric_limits<double>::infinity();
  double best_sq = std::numeric_limits<double>::infinity();
  for (const double* p : B) {
    double s = squared_l2(q, p, dim);
    if (s < best_sq) best_sq = s;
  }
  return std::sqrt(best_sq);
}

inline double min_l2(const std::vector<double>& q, const std::vector<const double*>& B) {
  return min_l2(q.data(), static_cast<int>(q.size()), B);
}

// Owned-copy variant for harness-side window snapshots (QuadTree / Uniform).
class BruteForceL2 {
 public:
  void clear() { points_.clear(); }

  void insert(const std::vector<double>& p) { points_.push_back(p); }

  void build(const std::vector<std::vector<double>>& B) {
    points_ = B;
  }

  double query(const std::vector<double>& q) const {
    if (points_.empty())
      throw std::runtime_error("exact_nn::BruteForceL2: empty B");
    const int dim = static_cast<int>(q.size());
    double best_sq = std::numeric_limits<double>::infinity();
    for (const auto& p : points_) {
      double s = squared_l2(q.data(), p.data(), dim);
      if (s < best_sq) best_sq = s;
    }
    return std::sqrt(best_sq);
  }

  std::size_t size() const { return points_.size(); }

 private:
  std::vector<std::vector<double>> points_;
};

}  // namespace exact_nn
