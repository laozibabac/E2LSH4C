#pragma once
// Fast exact one-sided Dynamic Chamfer (sum_{a in A} min_{b in B} ||a-b||_2).
// API-compatible with vendor DynamicChamfer for ctor / insert_B / delete_B / current.
// Optimizations (exact semantics):
//   - Contiguous AoS flat A (and B slots) for distance loops
//   - Compare with squared distance; sqrt only when accepting a new minimum
//   - Default: full SIMD squared-L2 via simd_l2.hpp (AVX-512/AVX2/scalar)
//   - Optional early-exit scalar path behind E2LSH_NN_EARLY_EXIT (default OFF)
//   - Stable physical B slots + logical order_ so delete does not renumber nn indices
//   - OpenMP parallel over A for insert and delete recomputes

#include "simd_l2.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#ifndef FAST_CHAMFER_NPOS
static constexpr std::size_t FAST_CHAMFER_NPOS = static_cast<std::size_t>(-1);
#endif

class FastDynamicChamfer {
 public:
  explicit FastDynamicChamfer(std::size_t dim)
      : nA_(0), dim_(dim), sum_(0.0) {
    if (dim_ == 0) throw std::invalid_argument("dim must be positive.");
    reserve_a_slots_(64);
    reserve_slots_(64);
  }

  FastDynamicChamfer(const std::vector<std::vector<double>>& A_init,
                     const std::vector<std::vector<double>>& B_init)
      : nA_(0),
        dim_(A_init.empty() ? 0 : A_init[0].size()),
        sum_(0.0) {
    if (A_init.empty())
      throw std::invalid_argument("A cannot be empty.");
    dim_ = A_init[0].size();
    reserve_a_slots_(std::max<std::size_t>(A_init.size() * 2 + 8, 64));
    for (const auto& a : A_init) {
      if (a.size() != dim_)
        throw std::invalid_argument("Point dimensions are inconsistent.");
      (void)insert_A(a);
    }
    reserve_slots_(std::max<std::size_t>(B_init.size() * 2 + 8, 64));
    for (const auto& b : B_init) insert_B(b);
  }

  void insert_B(const std::vector<double>& b_new) {
    check_dim_(b_new);
    const std::size_t slot = alloc_slot_();
    double* bp = &B_flat_[slot * dim_];
    for (std::size_t d = 0; d < dim_; ++d) bp[d] = b_new[d];
    occupied_[slot] = 1;
    order_.push_back(slot);

    const double* Adata = A_flat_.data();
    const int dim = static_cast<int>(dim_);
    const std::size_t nA = a_order_.size();
    const std::size_t* aord = a_order_.data();

    double delta = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : delta)
#endif
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(nA); ++ii) {
      const std::size_t i = aord[static_cast<std::size_t>(ii)];
      const double* ap = Adata + i * static_cast<std::size_t>(dim);
      const double bound = min_sq_[i];
#if E2LSH_NN_EARLY_EXIT
      double sq = 0.0;
      int d = 0;
      for (; d < dim; ++d) {
        const double diff = ap[d] - bp[d];
        sq += diff * diff;
        if (sq >= bound) break;
      }
      if (d != dim) continue;  // early-rejected vs current best
#else
      const double sq = e2lsh::simd_l2::squared_l2(ap, bp, dim);
      if (sq >= bound) continue;
#endif
      // sq < bound (or bound was +inf)
      const double dist = std::sqrt(sq);
      const double old = min_dist_[i];
      if (old == std::numeric_limits<double>::infinity())
        delta += dist;
      else
        delta += (dist - old);
      min_dist_[i] = dist;
      min_sq_[i] = sq;
      nn_slot_[i] = slot;
    }
    sum_ += delta;
  }

  void delete_B(std::size_t idx) {
    if (idx >= order_.size())
      throw std::out_of_range("delete_B: Index out of bounds.");
    const std::size_t slot_del = order_[idx];
    order_.erase(order_.begin() + static_cast<std::ptrdiff_t>(idx));
    occupied_[slot_del] = 0;
    free_list_.push_back(slot_del);

    std::vector<std::size_t> to_recompute;
    to_recompute.reserve(a_order_.size() / 16 + 1);
    for (std::size_t i : a_order_) {
      if (nn_slot_[i] == slot_del) {
        sum_ -= min_dist_[i];
        min_dist_[i] = std::numeric_limits<double>::infinity();
        min_sq_[i] = std::numeric_limits<double>::infinity();
        nn_slot_[i] = FAST_CHAMFER_NPOS;
        to_recompute.push_back(i);
      }
    }

    if (to_recompute.empty() || order_.empty()) return;

    const double* Adata = A_flat_.data();
    const double* Bdata = B_flat_.data();
    const int dim = static_cast<int>(dim_);
    const std::size_t* order_ptr = order_.data();
    const std::size_t nB = order_.size();
    const std::size_t nR = to_recompute.size();

    double delta = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : delta)
#endif
    for (std::ptrdiff_t rr = 0; rr < static_cast<std::ptrdiff_t>(nR); ++rr) {
      const std::size_t i = to_recompute[static_cast<std::size_t>(rr)];
      const double* ap = Adata + i * static_cast<std::size_t>(dim);
      double best_sq = std::numeric_limits<double>::infinity();
      std::size_t best_slot = FAST_CHAMFER_NPOS;
      for (std::size_t j = 0; j < nB; ++j) {
        const std::size_t slot = order_ptr[j];
        const double* bp = Bdata + slot * static_cast<std::size_t>(dim);
#if E2LSH_NN_EARLY_EXIT
        double sq = 0.0;
        int d = 0;
        for (; d < dim; ++d) {
          const double diff = ap[d] - bp[d];
          sq += diff * diff;
          if (sq >= best_sq) break;
        }
        if (d != dim) continue;
        best_sq = sq;
        best_slot = slot;
#else
        const double sq = e2lsh::simd_l2::squared_l2(ap, bp, dim);
        if (sq < best_sq) {
          best_sq = sq;
          best_slot = slot;
        }
#endif
      }
      const double dist = std::sqrt(best_sq);
      min_dist_[i] = dist;
      min_sq_[i] = best_sq;
      nn_slot_[i] = best_slot;
      delta += dist;
    }
    sum_ += delta;
  }

  std::size_t insert_A(const std::vector<double>& a_new) {
    if (dim_ == 0) dim_ = a_new.size();
    check_dim_(a_new);
    const std::size_t slot = alloc_a_slot_();
    double* ap = &A_flat_[slot * dim_];
    for (std::size_t d = 0; d < dim_; ++d) ap[d] = a_new[d];
    a_occupied_[slot] = 1;
    a_order_.push_back(slot);
    nA_ = a_order_.size();

    double best_sq = std::numeric_limits<double>::infinity();
    std::size_t best_slot = FAST_CHAMFER_NPOS;
    const double* Bdata = B_flat_.data();
    const int dim = static_cast<int>(dim_);
    for (std::size_t j = 0; j < order_.size(); ++j) {
      const std::size_t bslot = order_[j];
      const double* bp = Bdata + bslot * dim_;
      const double sq = e2lsh::simd_l2::squared_l2(ap, bp, dim);
      if (sq < best_sq) {
        best_sq = sq;
        best_slot = bslot;
      }
    }
    const double dist = (best_slot == FAST_CHAMFER_NPOS)
                            ? std::numeric_limits<double>::infinity()
                            : std::sqrt(best_sq);
    min_dist_[slot] = dist;
    min_sq_[slot] = best_sq;
    nn_slot_[slot] = best_slot;
    if (best_slot != FAST_CHAMFER_NPOS) sum_ += dist;
    return slot;
  }

  void delete_A(std::size_t idx) {
    if (idx >= a_order_.size())
      throw std::out_of_range("delete_A: Index out of bounds.");
    const std::size_t slot = a_order_[idx];
    if (min_dist_[slot] != std::numeric_limits<double>::infinity())
      sum_ -= min_dist_[slot];
    a_order_.erase(a_order_.begin() + static_cast<std::ptrdiff_t>(idx));
    a_occupied_[slot] = 0;
    a_free_list_.push_back(slot);
    nA_ = a_order_.size();
    min_dist_[slot] = std::numeric_limits<double>::infinity();
    min_sq_[slot] = std::numeric_limits<double>::infinity();
    nn_slot_[slot] = FAST_CHAMFER_NPOS;
  }

  double current() const { return sum_; }

  std::size_t size_B() const { return order_.size(); }
  std::size_t size_A() const { return a_order_.size(); }

  std::vector<double> liveMinDists() const {
    std::vector<double> out;
    out.reserve(a_order_.size());
    for (std::size_t s : a_order_) out.push_back(min_dist_[s]);
    return out;
  }

 private:
  std::size_t nA_ = 0;
  std::size_t dim_ = 0;
  std::vector<double, e2lsh::AlignedAlloc<double>> A_flat_;
  std::vector<unsigned char> a_occupied_;
  std::vector<std::size_t> a_order_;
  std::vector<std::size_t> a_free_list_;
  std::size_t n_a_slots_ = 0;

  std::vector<double, e2lsh::AlignedAlloc<double>> B_flat_;
  std::vector<unsigned char> occupied_;
  std::vector<std::size_t> order_;
  std::vector<std::size_t> free_list_;
  std::size_t n_slots_ = 0;

  std::vector<double> min_dist_;
  std::vector<double> min_sq_;
  std::vector<std::size_t> nn_slot_;
  double sum_ = 0.0;

  void check_dim_(const std::vector<double>& p) const {
    if (p.size() != dim_)
      throw std::invalid_argument("Insertion point dimensions are inconsistent.");
  }

  void reserve_slots_(std::size_t cap) {
    if (cap <= n_slots_) return;
    B_flat_.resize(cap * dim_, 0.0);
    occupied_.resize(cap, 0);
    for (std::size_t s = n_slots_; s < cap; ++s) free_list_.push_back(s);
    n_slots_ = cap;
  }

  std::size_t alloc_slot_() {
    if (free_list_.empty()) reserve_slots_(std::max<std::size_t>(n_slots_ * 2, 64));
    std::size_t slot = free_list_.back();
    free_list_.pop_back();
    return slot;
  }

  void reserve_a_slots_(std::size_t cap) {
    if (cap <= n_a_slots_) return;
    A_flat_.resize(cap * dim_, 0.0);
    a_occupied_.resize(cap, 0);
    min_dist_.resize(cap, std::numeric_limits<double>::infinity());
    min_sq_.resize(cap, std::numeric_limits<double>::infinity());
    nn_slot_.resize(cap, FAST_CHAMFER_NPOS);
    for (std::size_t s = n_a_slots_; s < cap; ++s) a_free_list_.push_back(s);
    n_a_slots_ = cap;
  }

  std::size_t alloc_a_slot_() {
    if (a_free_list_.empty()) reserve_a_slots_(std::max<std::size_t>(n_a_slots_ * 2, 64));
    std::size_t slot = a_free_list_.back();
    a_free_list_.pop_back();
    return slot;
  }
};
