#pragma once
// In-memory ℓ2 E2LSH Chamfer estimator.
// queryStratified gives every nonempty first-collision layer one sample, then
// apportions the remaining budget by largest remainder. Within a layer it
// samples uniformly, or enumerates the layer when the budget covers it.
// queryImportance draws each sample independently with probability proportional
// to its first-collision weight. It is only the ablation estimator.
#include "e2lsh_hash.hpp"
#include "weighted_sampler.hpp"
#include "exact_nn.hpp"
#include "simd_l2.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstring>
#include <deque>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace e2lsh {

// CoordKey defines memcmp equality. The insert path keys exact cells by
// hashCoords and FingerMap, not by CoordKey.
struct CoordKey {
  const double* p = nullptr;
  int d = 0;
  std::size_t hash = 0;
  std::size_t hash2 = 0;

  void rehash() {
    hashCoords(p, d, hash, hash2);
  }

  bool operator==(const CoordKey& o) const {
    if (d != o.d) return false;
    if (hash != o.hash || hash2 != o.hash2) return false;
    if (p == o.p) return true;
    if (d == 0) return true;
    return std::memcmp(p, o.p, static_cast<size_t>(d) * sizeof(double)) == 0;
  }
};

struct CoordKeyHash {
  std::size_t operator()(const CoordKey& c) const noexcept { return c.hash; }
};

struct ExactCell;
struct Bucket;

struct AMembership {
  Bucket* bucket = nullptr;
  size_t pos = 0;
};

// 一个还活着的 A 点。坐标住在 A_flat_，精确格子住在 cell。
// 它所在的哈希路径不存成数组，而是 cell->l0_bucket 再沿 parent 往上爬。
struct ARecord {
  PointId id = 0;
  size_t coord_off = 0;  // 在 A_flat_ 里的起点，每次跳 dim_ 个 float
  ExactCell* cell = nullptr;
  size_t exact_pos = 0;  // 自己在 cell->A_members 里的下标，删除时 O(1) 换尾
};

// 一个还活着的 B 点。membership[i] 缓存它在第 i 层的桶指针，删 B 时不用重新哈希。
struct BRecord {
  PointId id = 0;
  ExactCell* cell = nullptr;
  std::vector<Bucket*> membership;  // 长度 L+1
  size_t nn_pos = 0;  // 在 nn_B_ / B_flat_ 里的位置，精确 NN 扫的就是这份
};

// Exact-coordinate leaf, layer -1. One cell holds one coordinate string.
// B_count > 0 means D_a = 0, so those A points enter no stratum.
// Equality uses the two fingerprints, not a full coordinate compare.
struct ExactCell {
  uint32_t id = 0;
  size_t coord_off = 0;
  uint64_t coord_hash = 0;
  uint64_t coord_hash2 = 0;
  int A_count = 0;
  int B_count = 0;
  std::vector<PointId> A_members;
  // Empty B here and nonempty B in the L0 bucket: the cell's A points belong to S_0.
  Bucket* l0_bucket = nullptr;
  SamplerSlotRef stratum_slot;
  bool in_stratum = false;
  int stratum_idx = -1;  // 进了 S_0 就是 0，否则 -1
  int stratum_dense_pos = -1;
  uint64_t sync_stamp = 0;  // L0 的 B 翻转时，避免同一格被扫两遍
};

// Hash bucket, layers 0..L. parent is coarser; children are finer.
// There is no layer-L+1 root node and no second subtree-weight index.
// A bucket enters its parent's stratum when its own B is empty and the parent's B is not.
struct Bucket {
  uint32_t id = 0;
  int layer = 0;
  // Dictionary key is the two fingerprints. The full K-tuple key_h stays only
  // on the current coarsest layer; colder buckets drop it.
  std::size_t key_hash = 0;
  std::size_t key_hash2 = 0;
  std::vector<int64_t> key_h;
  Bucket* parent = nullptr;
  int A_count = 0;  // A points in the subtree
  int B_count = 0;  // occupied when B_count > 0
  std::vector<PointId> A_members;  // member list only at layer 0
  std::vector<Bucket*> children;
  std::vector<int> child_weight;  // children[i]->A_count; sampling scans this array
  int child_pos = -1;
  SamplerSlotRef stratum_slot;
  bool in_stratum = false;
  int stratum_idx = -1;  // 0..L is S_i, L+1 is the fallback, -1 is in no stratum
  int stratum_dense_pos = -1;
};

inline double l2(const double* a, const double* b, int dim) {
  return std::sqrt(simd_l2::squared_l2(a, b, dim));
}
inline double l2(const std::vector<double>& a, const std::vector<double>& b) {
  // SIMD squared-L2 then sqrt (used by updateRangeAndExpand / insertABatch R_max).
  return l2(a.data(), b.data(), static_cast<int>(a.size()));
}

struct ProfileBreakdown {
  // insertA (ms totals); expandTop nested time is excluded from insertA_* and reported separately
  double insertA_h0_ms = 0.0;
  double insertA_hash_ms = 0.0;
  double insertA_exact_ms = 0.0;
  double insertA_map_ms = 0.0;
  double insertA_coord_io_ms = 0.0;
  double insertA_posting_ms = 0.0;
  double insertA_resolve_ms = 0.0;
  double insertA_sampler_ms = 0.0;
  double insertA_other_ms = 0.0;
  int64_t insertA_count = 0;

  // insertB
  double insertB_h0_ms = 0.0;
  double insertB_resolve_ms = 0.0;
  double insertB_exact_sampler_ms = 0.0;
  double insertB_other_ms = 0.0;
  int64_t insertB_count = 0;

  // deleteB
  double deleteB_membership_ms = 0.0;
  double deleteB_exact_sampler_ms = 0.0;
  double deleteB_other_ms = 0.0;
  int64_t deleteB_count = 0;

  // expandTop cumulative (called from insertA/insertB via updateRangeAndExpand)
  double expandTop_ms = 0.0;
  int64_t expandTop_count = 0;

  // query per-sample buckets
  double query_sample_ms = 0.0;   // Fenwick sample + pick A from component
  double query_da_ms = 0.0;       // binary-search first collision / recover Da
  double query_nn_ms = 0.0;       // exact NN da
  double query_overhead_ms = 0.0; // form X / loop / setup
  double query_io_ms = 0.0;       // out-of-core posting+coord reads (0 for in-RAM)
  int64_t query_calls = 0;
  int64_t query_samples = 0;
  int64_t io_reads = 0;
  int64_t io_writes = 0;
  int64_t io_bytes_read = 0;

  double insertA_total_ms() const {
    return insertA_h0_ms + insertA_hash_ms + insertA_exact_ms + insertA_map_ms + insertA_coord_io_ms +
           insertA_posting_ms + insertA_resolve_ms + insertA_sampler_ms + insertA_other_ms;
  }
  double insertB_total_ms() const {
    return insertB_h0_ms + insertB_resolve_ms + insertB_exact_sampler_ms + insertB_other_ms;
  }
  double deleteB_total_ms() const {
    return deleteB_membership_ms + deleteB_exact_sampler_ms + deleteB_other_ms;
  }
  double query_total_ms() const {
    return query_sample_ms + query_da_ms + query_nn_ms + query_overhead_ms + query_io_ms;
  }
};


// 分层预算体检单：不跑最近邻，只看「这一层分到几个名额、要不要整班点名」。
struct AllocationStats {
  int T = 0;
  int L = 0;
  int n_strata = 0;
  int n_nonempty = 0;       // Ni > 0
  int n_small = 0;          // Ti >= Ni > 0 (full-enumerated strata)
  int actual_samples = 0;   // sum_i min(Ti, Ni)
  int leftover = 0;         // T - actual_samples
  std::vector<int> N;
  std::vector<double> W;
  std::vector<int> Ti;
  std::vector<char> flag_small;  // 1 iff Ti >= Ni > 0
};

// Largest-remainder apportionment. Each layer receives floor(T * W_i / sum W).
// Each leftover seat goes to one layer, largest fractional part first.
// Equal fractions go to the smaller index.
inline std::vector<int> hamiltonAllocate(const std::vector<double>& W, int T) {
  const int n = static_cast<int>(W.size());
  std::vector<int> ti(static_cast<size_t>(n), 0);
  if (T <= 0 || n == 0) return ti;
  double Wsum = 0.0;
  for (double w : W) Wsum += w;
  if (!(Wsum > 0.0)) return ti;
  std::vector<std::pair<double, int>> frac;
  frac.reserve(static_cast<size_t>(n));
  int allocated = 0;
  for (int i = 0; i < n; ++i) {
    const double q = static_cast<double>(T) * W[static_cast<size_t>(i)] / Wsum;
    const int base = static_cast<int>(std::floor(q));
    ti[static_cast<size_t>(i)] = base;
    allocated += base;
    frac.emplace_back(q - static_cast<double>(base), i);
  }
  const int rem = T - allocated;
  std::sort(frac.begin(), frac.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) return a.first > b.first;
    return a.second < b.second;
  });
  for (int k = 0; k < rem && k < n; ++k) {
    ti[static_cast<size_t>(frac[static_cast<size_t>(k)].second)] += 1;
  }
  return ti;
}

// Nonempty layers each receive one seat, then hamiltonAllocate splits T - s.
// Empty layers stay at 0. T < s is rejected. T == s leaves every nonempty layer at 1.
inline std::vector<int> hamiltonAllocateGuaranteed(const std::vector<double>& W,
                                                   const std::vector<int>& N,
                                                   int T) {
  const int n = static_cast<int>(W.size());
  std::vector<int> Ti(static_cast<size_t>(n), 0);
  if (n == 0) return Ti;
  if (static_cast<int>(N.size()) != n) {
    throw std::invalid_argument("hamiltonAllocateGuaranteed: W and N size mismatch");
  }
  std::vector<int> active;
  active.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    if (N[static_cast<size_t>(i)] > 0) active.push_back(i);
  }
  const int nonempty = static_cast<int>(active.size());
  if (nonempty == 0) return Ti;
  if (T < nonempty) {
    throw std::invalid_argument(
        "stratified sample budget T must be >= number of non-empty strata");
  }
  for (int idx : active) Ti[static_cast<size_t>(idx)] = 1;
  const int extra_budget = T - nonempty;
  if (extra_budget <= 0) return Ti;

  std::vector<double> packed(static_cast<size_t>(nonempty), 0.0);
  for (int j = 0; j < nonempty; ++j) {
    packed[static_cast<size_t>(j)] = W[static_cast<size_t>(active[static_cast<size_t>(j)])];
  }
  const std::vector<int> extra = hamiltonAllocate(packed, extra_budget);
  for (int j = 0; j < nonempty; ++j) {
    Ti[static_cast<size_t>(active[static_cast<size_t>(j)])] += extra[static_cast<size_t>(j)];
  }
  return Ti;
}

// 按层权重抽一层。W 里至少有一个正数，D = Σ W。uniform_real 取 [0, D)。
// 浮点把累加和推过 u 时落到最后一个正权重层。
inline int drawStratumProportional(const std::vector<double>& W, double D, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> uni(0.0, D);
  const double u = uni(rng);
  double acc = 0.0;
  int last = -1;
  const int n = static_cast<int>(W.size());
  for (int i = 0; i < n; ++i) {
    const double w = W[static_cast<size_t>(i)];
    if (!(w > 0.0)) continue;
    last = i;
    acc += w;
    if (u < acc) return i;
  }
  if (last < 0) throw std::runtime_error("drawStratumProportional: empty mass");
  return last;
}

// 动态维护 A、B 两套点，随时回答「现在的倒角距离大概是多少」。
// 默认路径是 queryStratified。queryImportance 是消融用的按 D_a 抽样，不进默认查询。
class ChamferEstimator {
 public:
  // rho is the minimum positive-distance bound Δ_0. It is an input, not a scanned minimum.
  // n_max is the point bound used to choose K. seed draws projections, shifts, and samples.
  ChamferEstimator(int dim, double rho, int n_max, uint64_t seed = 42, int k_override = -1)
      : dim_(dim),
        rho_(rho),
        n_max_(n_max),
        K_(k_override > 0 ? k_override : computeK(n_max)),
        rng_(seed) {
    if (!(rho_ > 0.0)) throw std::invalid_argument("rho must be > 0");
    if (dim_ <= 0) throw std::invalid_argument("dim must be > 0");
    if (K_ <= 0) throw std::invalid_argument("K must be > 0");
    // K independent directions g_j ~ N(0, I_d), stored row-major.
    std::normal_distribution<double> normal(0.0, 1.0);
    g_flat_.assign(static_cast<size_t>(K_) * static_cast<size_t>(dim_), 0.0);
    // 坐标按 float32 预留整块。GIST 的 f32bin 本来就是 float，f32→f64→f32 位型不变。
    // 一次留够，避免 double 池扩到 8 GiB 时再拷一份把页面挤进压缩内存。
    A_flat_.reserve(static_cast<std::size_t>(n_max_) * static_cast<std::size_t>(dim_));
    for (int j = 0; j < K_; ++j)
      for (int i = 0; i < dim_; ++i) g_flat_[static_cast<size_t>(j) * dim_ + i] = normal(rng_);
    // Finest-layer shift only: Unif[0, K ρ). Insert and delete do not redraw g_j.
    // Each new level draws K bits ξ.
    double w0 = static_cast<double>(K_) * rho_;
    std::uniform_real_distribution<double> uni(0.0, w0);
    S0_.resize(K_);
    for (int j = 0; j < K_; ++j) S0_[j] = uni(rng_);
    L_ = 0;  // levels grow upward from 0; there is no prebuilt root
    deltas_.push_back(rho_);  // Δ_0 = ρ
    layer_maps_.resize(1);
    buckets_.resize(1);  // 下标 0 空着，桶 id 从 1 起
    exact_by_id_.resize(1);
    xi_.clear();
    has_anchor_ = false;
    R_seen_ = 0.0;
    stratum_samp_.assign(2, DynamicWeightedSampler{});  // S_0 和 fallback S_{L+1}，L 变了再扩
    stratum_N_.assign(2, 0);
    stratum_active_.assign(2, {});
  }

  int K() const { return K_; }
  int L() const { return L_; }
  double rho() const { return rho_; }
  std::size_t sizeA() const { return a_live_; }
  std::size_t sizeB() const { return b_live_; }
  const double* nnBData() const { return B_flat_.data(); }
  double totalD() const {
    double D = 0.0;
    const int nAvail = static_cast<int>(stratum_N_.size());
    for (int i = 0; i <= L_ && i < nAvail; ++i) {
      D += static_cast<double>(stratum_N_[static_cast<size_t>(i)]) * deltas_[static_cast<size_t>(i)];
    }
    if (L_ + 1 < nAvail) {
      D += static_cast<double>(stratum_N_[static_cast<size_t>(L_ + 1)]) * 2.0 *
           deltas_[static_cast<size_t>(L_)];
    }
    return D;
  }

  const ProfileBreakdown& profile() const { return prof_; }
  void resetProfile() { prof_ = ProfileBreakdown{}; }

  // Insert one A point: hash it, complete the path, and add one to each A_count.
  // B counts stay put. Only the frontier bucket on this path is refreshed.
  PointId insertA(const std::vector<double>& x) {
    using clock = std::chrono::high_resolution_clock;
    ensureDim(x);
    ensureCapacityA();
    double e0 = prof_.expandTop_ms;
    auto t0 = clock::now();
    updateRangeAndExpand(x);  // 点太远就把楼盖高，直到 Δ_L ≥ 2 R_seen
    auto t1 = clock::now();
    double expand_part = prof_.expandTop_ms - e0;
    prof_.insertA_other_ms += msBetween(t0, t1) - expand_part;

    t0 = clock::now();
    fillH0(x.data(), h0_scratch_);  // 只算最细层 H_0，更粗的层靠 parent 指针爬
    t1 = clock::now();
    prof_.insertA_h0_ms += msBetween(t0, t1);

    return insertAStructure(x, h0_scratch_);
  }

  // Batch insert: pre-expand to R_max (same expandTop count / xi_ as sequential
  // insertA over pts), parallel H0, then serial structure inserts (maps not thread-safe).
  std::vector<PointId> insertABatch(const std::vector<std::vector<double>>& pts) {
    using clock = std::chrono::high_resolution_clock;
    if (pts.empty()) return {};
    const int n = static_cast<int>(pts.size());
    for (const auto& p : pts) ensureDim(p);
    if (static_cast<int>(a_live_) + n > n_max_)
      throw std::runtime_error("|A| would exceed n_max");

    // 1) Range + expand to completion before any insert (empty expandTop).
    auto t0 = clock::now();
    if (!has_anchor_) {
      anchor_ = pts[0];
      has_anchor_ = true;
      R_seen_ = 0.0;
    }
    double R_max = R_seen_;
#if defined(_OPENMP)
    if (n >= 64) {
#pragma omp parallel for reduction(max : R_max) schedule(static)
      for (int i = 0; i < n; ++i) {
        R_max = std::max(R_max, l2(pts[static_cast<size_t>(i)], anchor_));
      }
    } else
#endif
    {
      for (int i = 0; i < n; ++i)
        R_max = std::max(R_max, l2(pts[static_cast<size_t>(i)], anchor_));
    }
    auto t1 = clock::now();
    prof_.insertA_other_ms += msBetween(t0, t1);

    while (deltas_[L_] < 2.0 * R_max - 1e-15) {
      expandTop();
    }
    R_seen_ = R_max;

    // 2) Parallel H0 (read-only g_flat_ / S0_ / K_ / rho_). Directly from pts[i],
    // no packed X: that copy was a full n×dim memset+memcpy and killed L1 reuse of x_i.
    std::vector<int64_t> h_pack(static_cast<size_t>(n) * static_cast<size_t>(K_));
    std::vector<uint64_t> hh1(static_cast<size_t>(n)), hh2(static_cast<size_t>(n));
    t0 = clock::now();
    fillH0BatchRows([&](int i) { return pts[static_cast<size_t>(i)].data(); }, n, dim_, K_,
                    g_flat_.data(), S0_.data(), rho_, h_pack.data(), hh1.data(), hh2.data());
    t1 = clock::now();
    prof_.insertA_h0_ms += msBetween(t0, t1);

    if (n < 64) {
      std::vector<PointId> ids;
      ids.reserve(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) {
        BucketKey key;
        key.setK(K_);
        key.h.assign(h_pack.begin() + static_cast<size_t>(i) * static_cast<size_t>(K_),
                     h_pack.begin() + static_cast<size_t>(i + 1) * static_cast<size_t>(K_));
        key.hash = static_cast<std::size_t>(hh1[static_cast<size_t>(i)]);
        key.hash2 = static_cast<std::size_t>(hh2[static_cast<size_t>(i)]);
        ids.push_back(insertAStructure(pts[static_cast<size_t>(i)], key));
      }
      return ids;
    }

    std::vector<std::size_t> ch1(static_cast<size_t>(n)), ch2(static_cast<size_t>(n));
    t0 = clock::now();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
      hashCoords(pts[static_cast<size_t>(i)].data(), dim_, ch1[static_cast<size_t>(i)],
                 ch2[static_cast<size_t>(i)]);
#else
    for (int i = 0; i < n; ++i)
      hashCoords(pts[static_cast<size_t>(i)].data(), dim_, ch1[static_cast<size_t>(i)],
                 ch2[static_cast<size_t>(i)]);
#endif
    t1 = clock::now();
    prof_.insertA_hash_ms += msBetween(t0, t1);

    std::vector<ExactCell*> cells(static_cast<size_t>(n), nullptr);
    std::vector<size_t> coff(static_cast<size_t>(n), 0);
    t0 = clock::now();
    for (int i = 0; i < n; ++i) {
      cells[static_cast<size_t>(i)] = getOrCreateExactHashed(
          pts[static_cast<size_t>(i)].data(), ch1[static_cast<size_t>(i)], ch2[static_cast<size_t>(i)],
          &coff[static_cast<size_t>(i)]);
    }
    t1 = clock::now();
    prof_.insertA_exact_ms += msBetween(t0, t1);

    const int nlay = L_ + 1;
    std::vector<Bucket*> path_pack(static_cast<size_t>(n) * static_cast<size_t>(nlay), nullptr);
    t0 = clock::now();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
      internLayerPathLocked(h_pack.data() + static_cast<size_t>(i) * static_cast<size_t>(K_),
                            hh1[static_cast<size_t>(i)], hh2[static_cast<size_t>(i)],
                            path_pack.data() + static_cast<size_t>(i) * static_cast<size_t>(nlay));
    }
    t1 = clock::now();
    prof_.insertA_resolve_ms += msBetween(t0, t1);

    t0 = clock::now();
    rebuildRamChildLists();
    t1 = clock::now();
    prof_.insertA_other_ms += msBetween(t0, t1);

    std::vector<PointId> ids;
    ids.reserve(static_cast<size_t>(n));
    t0 = clock::now();
    for (int i = 0; i < n; ++i) {
      ids.push_back(insertAMembers(
          pts[static_cast<size_t>(i)], cells[static_cast<size_t>(i)], coff[static_cast<size_t>(i)],
          path_pack.data() + static_cast<size_t>(i) * static_cast<size_t>(nlay)));
    }
    t1 = clock::now();
    prof_.insertA_exact_ms += msBetween(t0, t1);

    t0 = clock::now();
    rebuildStrata();
    t1 = clock::now();
    prof_.insertA_sampler_ms += msBetween(t0, t1);
    return ids;
  }

  // 插入的镜像：沿精确叶子到顶层把 A_count 减 1，空节点回收。η = -1。
  void deleteA(PointId id) {
    ARecord* rec = aRec(id);

    // exact
    ExactCell* cell = rec->cell;
    {
      PointId moved = removeFromMemberListAt(cell->A_members, rec->exact_pos);
      if (moved != 0) aRec(moved)->exact_pos = rec->exact_pos;
    }
    cell->A_count -= 1;

    // Capture path via L0 + parent links; remove id from each A_members (scan).
    std::vector<Bucket*> path(static_cast<size_t>(L_) + 1, nullptr);
    Bucket* cur = cell->l0_bucket;
    if (!cur) throw std::runtime_error("deleteA: missing l0_bucket");
    for (int i = 0; i <= L_; ++i) {
      if (!cur) throw std::runtime_error("deleteA: broken parent chain");
      path[static_cast<size_t>(i)] = cur;
      cur = cur->parent;
    }

    // layers: A_members only on L0
    for (int i = 0; i <= L_; ++i) {
      Bucket* buck = path[static_cast<size_t>(i)];
      if (i == 0) {
        auto& mem = buck->A_members;
        auto it = std::find(mem.begin(), mem.end(), id);
        if (it == mem.end()) throw std::runtime_error("deleteA: id not in L0 members");
        size_t pos = static_cast<size_t>(it - mem.begin());
        (void)removeFromMemberListAt(mem, pos);
      }
      buck->A_count -= 1;
      publishChildWeight(buck);
    }
    // 分层：精确格可能离开 S_0；整条路上只需要同步「第一个有 B 的桶」下面那个前沿孩子。
    syncExactStratum(cell);
    syncPathFrontierBucket(path);

    maybeCleanupExact(cell);
    for (int i = 0; i <= L_; ++i) {
      Bucket* buck = path[static_cast<size_t>(i)];
      maybeCleanupBucket(i, buck);
    }

    a_by_id_[id] = nullptr;
    --a_live_;
    // record remains in a_pool_ (stable addresses); slot not reused (A mostly static)
  }

  // Insert one B point. A bucket whose B count crosses from 0 to 1 moves the
  // first-collision boundary and syncs its children. Layer 0 also scans cells.
  // B may be empty; queryStratified then returns infinity.
  PointId insertB(const std::vector<double>& x) {
    using clock = std::chrono::high_resolution_clock;
    ensureDim(x);
    ensureCapacityB();
    double e0 = prof_.expandTop_ms;
    auto t0 = clock::now();
    updateRangeAndExpand(x);
    auto t1 = clock::now();
    double expand_part = prof_.expandTop_ms - e0;
    prof_.insertB_other_ms += msBetween(t0, t1) - expand_part;

    PointId id = next_b_id_++;
    BRecord* rec = allocB();
    rec->id = id;
    if (b_by_id_.size() <= id) b_by_id_.resize(id + 1, nullptr);
    b_by_id_[id] = rec;
    ++b_live_;

    t0 = clock::now();
    fillH0(x.data(), h0_scratch_);
    t1 = clock::now();
    prof_.insertB_h0_ms += msBetween(t0, t1);

    // exact: if B_count 0->1, zero out mass
    t0 = clock::now();
    ExactCell* cell = getOrCreateExact(x);
    rec->cell = cell;
    bool exact_b_zero = (cell->B_count == 0);  // 即将 0→1：这个格子里的 A 点 D_a 变成 0，退出 S_0
    cell->B_count += 1;
    if (exact_b_zero) syncExactStratum(cell);
    t1 = clock::now();
    prof_.insertB_exact_sampler_ms += msBetween(t0, t1);

    // layers — same resolveLayerPath as insertA; cache Bucket* for O(L) deleteB
    rec->membership.resize(L_ + 1);
    t0 = clock::now();
    resolveLayerPath(h0_scratch_);
    t1 = clock::now();
    prof_.insertB_resolve_ms += msBetween(t0, t1);
    const std::vector<Bucket*>& path = resolve_path_;
    t0 = clock::now();
    for (int i = 0; i <= L_; ++i) {
      Bucket* buck = path[i];
      rec->membership[i] = buck;
      bool was_zero = (buck->B_count == 0);
      buck->B_count += 1;
      if (was_zero) onBucketBTransition(buck);  // 占用翻转：前沿从「自己」交到「孩子们」
    }
    chillPathKeys(path);
    t1 = clock::now();
    prof_.insertB_exact_sampler_ms += msBetween(t0, t1);

    // Append coordinates to the contiguous array scanned by exact NN.
    t0 = clock::now();
    rec->nn_pos = nn_B_.size();
    nn_B_.push_back(id);
    b_nn_index_[id] = rec->nn_pos;
    {
      size_t off = B_flat_.size();
      B_flat_.resize(off + static_cast<size_t>(dim_));
      for (int d = 0; d < dim_; ++d) B_flat_[off + static_cast<size_t>(d)] = x[static_cast<size_t>(d)];
    }
    t1 = clock::now();
    prof_.insertB_other_ms += msBetween(t0, t1);

    ++prof_.insertB_count;
    return id;
  }

  // B 删除。占用 1→0 时前沿交回去；空桶回收。精确 NN 数组做换尾删除。
  void deleteB(PointId id) {
    using clock = std::chrono::high_resolution_clock;
    BRecord* rec = bRec(id);

    auto t0 = clock::now();
    ExactCell* cell = rec->cell;
    if (!cell) throw std::runtime_error("deleteB: missing exact cell");
    cell->B_count -= 1;
    if (cell->B_count == 0) {
      syncExactStratum(cell);
    }
    maybeCleanupExact(cell);
    auto t1 = clock::now();
    prof_.deleteB_exact_sampler_ms += msBetween(t0, t1);

    // O(L) via cached membership pointers — no H0/parentKey/HashMap walk
    t0 = clock::now();
    if (static_cast<int>(rec->membership.size()) != L_ + 1)
      throw std::runtime_error("deleteB: stale B membership");
    for (int i = 0; i <= L_; ++i) {
      Bucket* buck = rec->membership[i];
      if (!buck) throw std::runtime_error("deleteB: missing bucket");
      buck->B_count -= 1;
      if (buck->B_count == 0) {
        onBucketBTransition(buck);
      }
      maybeCleanupBucket(i, buck);
    }
    t1 = clock::now();
    prof_.deleteB_membership_ms += msBetween(t0, t1);

    // NN swap-delete whole dim_ block in B_flat_
    t0 = clock::now();
    size_t pos = rec->nn_pos;
    size_t last = nn_B_.size() - 1;
    if (pos != last) {
      PointId moved = nn_B_[last];
      nn_B_[pos] = moved;
      const size_t D = static_cast<size_t>(dim_);
      double* dst = B_flat_.data() + pos * D;
      const double* src = B_flat_.data() + last * D;
      for (size_t d = 0; d < D; ++d) dst[d] = src[d];
      bRec(moved)->nn_pos = pos;
      b_nn_index_[moved] = pos;
    }
    nn_B_.pop_back();
    B_flat_.resize(nn_B_.size() * static_cast<size_t>(dim_));
    b_nn_index_.erase(id);

    b_by_id_[id] = nullptr;
    --b_live_;
    freeB(rec);
    t1 = clock::now();
    prof_.deleteB_other_ms += msBetween(t0, t1);

    ++prof_.deleteB_count;
  }

  // Every nonempty layer receives one sample. The remaining budget is split by
  // largest remainder. Within a layer the draw is uniform with replacement.
  // T_i >= N_i enumerates the layer. Distinct points are packed for exact NN.
  double queryStratified(int T = -1) {
    using clock = std::chrono::high_resolution_clock;
    auto t_setup0 = clock::now();
    // Empty A returns 0. Empty B with nonempty A returns infinity.
    if (a_live_ == 0) return 0.0;
    if (nn_B_.empty()) return std::numeric_limits<double>::infinity();
    if (T <= 0) T = defaultT(a_live_);  // T = floor(15 log2 |A|)

    const int nS = L_ + 2;  // S_0 .. S_L, then the fallback at index L+1
    ensureStratumSize();
    std::vector<double> W(static_cast<size_t>(nS), 0.0);
    std::vector<int> N(static_cast<size_t>(nS), 0);
    for (int i = 0; i <= L_; ++i) {
      N[static_cast<size_t>(i)] = stratum_N_[static_cast<size_t>(i)];  // N_i = |S_i|
      W[static_cast<size_t>(i)] = static_cast<double>(N[static_cast<size_t>(i)]) * deltas_[static_cast<size_t>(i)];  // W_i = N_i Δ_i
    }
    N[static_cast<size_t>(L_ + 1)] = stratum_N_[static_cast<size_t>(L_ + 1)];
    // No collision on a real layer: D_a = 2 Δ_L, so this layer's weight uses 2 Δ_L.
    W[static_cast<size_t>(L_ + 1)] =
        static_cast<double>(N[static_cast<size_t>(L_ + 1)]) * 2.0 * deltas_[static_cast<size_t>(L_)];

    double Dalloc = 0.0;
    for (double w : W) Dalloc += w;  // D = Σ W_i
    auto t_setup1 = clock::now();
    prof_.query_overhead_ms += msBetween(t_setup0, t_setup1);
    if (!(Dalloc > 0.0)) {
      ++prof_.query_calls;
      return 0.0;  // every A coincides with some B
    }

    std::vector<int> Ti = hamiltonAllocateGuaranteed(W, N, T);

    struct SampleTask { PointId aid; int multiplicity; };
    std::vector<SampleTask> tasks;
    tasks.reserve(static_cast<size_t>(T));
    // Per-stratum: draw list (sampling) or empty (full-enum uses tasks only)
    std::vector<std::vector<PointId>> draws(static_cast<size_t>(nS));
    std::vector<char> is_full(static_cast<size_t>(nS), 0);
    std::vector<int> task_lo(static_cast<size_t>(nS), 0);
    std::vector<int> task_hi(static_cast<size_t>(nS), 0);

    auto t_samp0 = clock::now();
    for (int si = 0; si < nS; ++si) {
      const int ni = N[static_cast<size_t>(si)];
      int ti = Ti[static_cast<size_t>(si)];
      if (ni <= 0 || ti <= 0) continue;
      task_lo[static_cast<size_t>(si)] = static_cast<int>(tasks.size());
      if (ti >= ni) {
        is_full[static_cast<size_t>(si)] = 1;
        collectStratumAIds(si, tasks);  // T_i >= N_i enumerates the layer
      } else {
        auto& dr = draws[static_cast<size_t>(si)];
        dr.resize(static_cast<size_t>(ti));
        std::unordered_map<PointId, int> mult;
        mult.reserve(static_cast<size_t>(ti));
        for (int t = 0; t < ti; ++t) {
          PointId aid = sampleAFromStratum(si);  // uniform with replacement
          dr[static_cast<size_t>(t)] = aid;
          ++mult[aid];
        }
        tasks.reserve(tasks.size() + mult.size());
        for (const auto& kv : mult) tasks.push_back(SampleTask{kv.first, kv.second});  // 同一点只算一次 NN
      }
      task_hi[static_cast<size_t>(si)] = static_cast<int>(tasks.size());
    }
    auto t_samp1 = clock::now();
    prof_.query_sample_ms += msBetween(t_samp0, t_samp1);

    auto t_nn0 = clock::now();
    std::vector<double> da_out;
    evalQueryNN(tasks, da_out);
    const int ntasks = static_cast<int>(da_out.size());
    auto t_nn1 = clock::now();
    prof_.query_nn_ms += msBetween(t_nn0, t_nn1);

    // Serial reduce (sampling walks draws in RNG order for identical FP sums)
    double Chat = 0.0;
    for (int si = 0; si < nS; ++si) {
      const int ni = N[static_cast<size_t>(si)];
      int ti = Ti[static_cast<size_t>(si)];
      if (ni <= 0 || ti <= 0) continue;
      const int lo = task_lo[static_cast<size_t>(si)];
      const int hi = task_hi[static_cast<size_t>(si)];
      if (is_full[static_cast<size_t>(si)]) {
        double sum_d = 0.0;
        for (int i = lo; i < hi; ++i) sum_d += da_out[static_cast<size_t>(i)];
        Chat += sum_d;  // 枚举层：Ĉ_i 就是这一层真实贡献 C_i
        prof_.query_samples += ni;
      } else {
        std::unordered_map<PointId, double> da_map;
        da_map.reserve(static_cast<size_t>(hi - lo));
        for (int i = lo; i < hi; ++i) {
          da_map[tasks[static_cast<size_t>(i)].aid] = da_out[static_cast<size_t>(i)];
        }
        double sum_d = 0.0;
        for (PointId aid : draws[static_cast<size_t>(si)]) sum_d += da_map[aid];  // 有放回，按抽到的顺序加
        Chat += (static_cast<double>(ni) / static_cast<double>(ti)) * sum_d;  // Ĉ_i = (N_i / T_i) Σ d
        prof_.query_samples += ti;
      }
    }
    ++prof_.query_calls;
    return Chat;
  }

  // 原始重要性采样。每次独立：以概率 W_i/D 选首碰层，再在该层均匀选点。
  // One draw is X = D * d_a / Δ_i. The fallback scale is 2 Δ_L. The return value is the mean of T draws.
  // 不预分配层预算，T_i ≥ N_i 时也不改成整层枚举。重复抽到的点只算一次最近邻。
  double queryImportance(int T = -1) {
    using clock = std::chrono::high_resolution_clock;
    auto t_setup0 = clock::now();
    if (a_live_ == 0) return 0.0;
    if (nn_B_.empty()) return std::numeric_limits<double>::infinity();
    if (T <= 0) T = defaultT(a_live_);

    std::vector<double> W;
    std::vector<double> scale;
    double D = 0.0;
    stratumImportanceWeights(W, scale, D);
    auto t_setup1 = clock::now();
    prof_.query_overhead_ms += msBetween(t_setup0, t_setup1);
    if (!(D > 0.0)) {
      ++prof_.query_calls;
      return 0.0;
    }

    struct Draw {
      PointId aid;
      int si;
    };
    std::vector<Draw> draws(static_cast<size_t>(T));
    auto t_samp0 = clock::now();
    std::unordered_map<PointId, int> mult;
    mult.reserve(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t) {
      const int si = drawStratumProportional(W, D, rng_);
      const PointId aid = sampleAFromStratum(si);
      draws[static_cast<size_t>(t)] = Draw{aid, si};
      ++mult[aid];
    }
    struct SampleTask {
      PointId aid;
      int multiplicity;
    };
    std::vector<SampleTask> tasks;
    tasks.reserve(mult.size());
    for (const auto& kv : mult) tasks.push_back(SampleTask{kv.first, kv.second});
    auto t_samp1 = clock::now();
    prof_.query_sample_ms += msBetween(t_samp0, t_samp1);

    auto t_nn0 = clock::now();
    std::vector<double> da_out;
    evalQueryNN(tasks, da_out);
    const int ntasks = static_cast<int>(da_out.size());
    auto t_nn1 = clock::now();
    prof_.query_nn_ms += msBetween(t_nn0, t_nn1);

    std::unordered_map<PointId, double> da_map;
    da_map.reserve(static_cast<size_t>(ntasks));
    for (int i = 0; i < ntasks; ++i)
      da_map[tasks[static_cast<size_t>(i)].aid] = da_out[static_cast<size_t>(i)];
    double sumX = 0.0;
    for (const Draw& dr : draws) {
      const double sc = scale[static_cast<size_t>(dr.si)];
      sumX += D * da_map[dr.aid] / sc;
    }
    prof_.query_samples += T;
    ++prof_.query_calls;
    return sumX / static_cast<double>(T);
  }

  // 只抽层号，概率 ∝ W_i。D = 0 时返回 -1。会推进 rng_。
  int sampleImportanceStratum() {
    std::vector<double> W;
    std::vector<double> scale;
    double D = 0.0;
    stratumImportanceWeights(W, scale, D);
    if (!(D > 0.0)) return -1;
    return drawStratumProportional(W, D, rng_);
  }

  // Expose last used T helper
  static int computeDefaultT(std::size_t n) { return defaultT(n); }

  // Debug: brute-force sum_a D_a
  double deltaAt(int i) const { return deltas_.at(i); }
  double Rseen() const { return R_seen_; }
  int numExactCells() const { return static_cast<int>(exact_pool_.size()); }

  // Per-point Da (same as computeDaConst)
  double daWeight(PointId id) const {
    return computeDaConst(aRec(id));
  }

  // First-collision layer i(a) = min{ i : H_i(a) meets B }.
  // -2: the exact cell already holds B (D_a = 0).
  // -1: no real layer collides, so D_a = 2 Δ_L.
  // 0..L: first collision at layer j, so D_a = Δ_j.
  int firstCollisionLayer(PointId id) const {
    const ARecord* arec = aRec(id);
    if (arec->cell->B_count > 0) return -2;
    Bucket* b = arec->cell->l0_bucket;
    for (int i = 0; i <= L_; ++i) {
      if (!b) return -1;
      if (b->B_count > 0) return i;
      b = b->parent;
    }
    return -1;
  }

  // 把这一行从 float 池展开成 double。返回的指针只活到下一次 coordsA。
  const double* coordsA(PointId id) const {
    coord_view_.resize(static_cast<std::size_t>(dim_));
    loadARow(aRec(id), coord_view_.data());
    return coord_view_.data();
  }
  std::vector<PointId> allAIds() const {
    std::vector<PointId> ids;
    ids.reserve(a_live_);
    for (PointId id = 1; id < a_by_id_.size(); ++id)
      if (a_by_id_[id]) ids.push_back(id);
    return ids;
  }

  double exactNNPublic(const double* x) const { return exactNN(x); }
  double exactNNPublic(const std::vector<double>& x) const { return exactNN(x.data()); }

  // Stratified diagnostics
  int stratumN(int si) const {
    ensureStratumSizeConst();
    if (si < 0 || si >= static_cast<int>(stratum_N_.size())) return 0;
    return stratum_N_[static_cast<size_t>(si)];
  }
  int stratumCount() const { return L_ + 2; }  // 0..L + fallback
  int fallbackStratumIndex() const { return L_ + 1; }
  // sum_i N_i over strata (positive-distance A only)
  int sumStratumN() const {
    ensureStratumSizeConst();
    int s = 0;
    for (int v : stratum_N_) s += v;
    return s;
  }
  int countPositiveDistanceA() const {
    int s = 0;
    for (PointId id = 1; id < a_by_id_.size(); ++id) {
      if (!a_by_id_[id]) continue;
      if (a_by_id_[id]->cell->B_count == 0) ++s;
    }
    return s;
  }
  // Each positive-distance A point belongs to exactly one S_i.
  bool validateStrataInvariant(double* sum_n_out = nullptr, double* pos_out = nullptr) const {
    int sn = sumStratumN();
    int pos = countPositiveDistanceA();
    if (sum_n_out) *sum_n_out = sn;
    if (pos_out) *pos_out = pos;
    return sn == pos;
  }
  // Snapshot RNG so IS vs stratified can share the same starting seed state.
  std::mt19937_64 rngSnapshot() const { return rng_; }
  void rngRestore(const std::mt19937_64& s) { rng_ = s; }
  PointId sampleStratumPublic(int si) { return sampleAFromStratum(si); }

  // Layer budgets only, no distances. Weights match queryStratified, including
  // the fallback weight 2 Δ_L. The allocation does not draw from rng_.
  AllocationStats allocationStats(int T = -1) {
    AllocationStats st;
    if (a_live_ == 0) return st;
    if (T <= 0) T = defaultT(a_live_);
    st.T = T;
    st.L = L_;
    const int nS = L_ + 2;
    st.n_strata = nS;
    ensureStratumSizeConst();
    st.N.assign(static_cast<size_t>(nS), 0);
    st.W.assign(static_cast<size_t>(nS), 0.0);
    for (int i = 0; i <= L_; ++i) {
      st.N[static_cast<size_t>(i)] = stratum_N_[static_cast<size_t>(i)];
      st.W[static_cast<size_t>(i)] =
          static_cast<double>(st.N[static_cast<size_t>(i)]) * deltas_[static_cast<size_t>(i)];
    }
    st.N[static_cast<size_t>(L_ + 1)] = stratum_N_[static_cast<size_t>(L_ + 1)];
    st.W[static_cast<size_t>(L_ + 1)] =
        static_cast<double>(st.N[static_cast<size_t>(L_ + 1)]) * 2.0 * deltas_[static_cast<size_t>(L_)];

    double Dalloc = 0.0;
    for (double w : st.W) Dalloc += w;
    if (!(Dalloc > 0.0)) {
      st.Ti.assign(static_cast<size_t>(nS), 0);
      st.flag_small.assign(static_cast<size_t>(nS), 0);
      st.leftover = T;
      return st;
    }

    st.Ti = hamiltonAllocateGuaranteed(st.W, st.N, T);
    st.flag_small.assign(static_cast<size_t>(nS), 0);
    for (int i = 0; i < nS; ++i) {
      const int ni = st.N[static_cast<size_t>(i)];
      const int ti = st.Ti[static_cast<size_t>(i)];
      if (ni > 0) ++st.n_nonempty;
      if (ni > 0 && ti >= ni) {
        ++st.n_small;
        st.flag_small[static_cast<size_t>(i)] = 1;
      }
      st.actual_samples += std::min(ti, ni);
    }
    st.leftover = T - st.actual_samples;
    return st;
  }


  // Debug: brute-force sum_a D_a
  double bruteSumDa() const {
    double s = 0.0;
    for (PointId id = 1; id < a_by_id_.size(); ++id)
      if (a_by_id_[id]) s += computeDaConst(a_by_id_[id]);
    return s;
  }

 private:
  static double msBetween(std::chrono::high_resolution_clock::time_point a,
                          std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  }

  ProfileBreakdown prof_;

  int dim_;
  double rho_;
  int n_max_;
  int K_;
  std::mt19937_64 rng_;

  std::vector<double> g_flat_;  // K 条高斯投影，行优先
  std::vector<double> S0_;      // 最细层平移，长度 K
  std::vector<std::vector<uint8_t>> xi_;  // K Bernoulli bits drawn when level i+1 opens
  int L_ = 0;                   // current coarsest real level; it grows with the data
  std::vector<double> deltas_;  // Δ_i = 2^i ρ

  bool has_anchor_ = false;
  std::vector<double> anchor_;
  double R_seen_ = 0.0;

  // One Fenwick sampler per layer. 0..L are S_i. L+1 is the fallback; there is no root node.
  // A slot's weight is that frontier component's A_count.
  mutable std::vector<DynamicWeightedSampler> stratum_samp_;
  mutable std::vector<int> stratum_N_;  // N_i，查询时直接读，不再扫点
  mutable std::vector<std::vector<ComponentId>> stratum_active_;  // 整班点名时按这个名单走
  mutable uint64_t sync_gen_ = 1;

  ShardedFingerMap exact_map_;
  std::deque<ExactCell> exact_pool_;
  std::vector<ExactCell*> exact_by_id_;  // [0] unused
  std::vector<ShardedFingerMap> layer_maps_;
  std::vector<Bucket*> buckets_;  // [0] unused
  std::deque<Bucket> bucket_pool_;
  std::vector<Bucket*> bucket_free_;
  std::mutex bucket_alloc_mu_;

  // Reused insert scratch (avoid per-insert BucketKey / path allocations).
  BucketKey h0_scratch_;
  std::vector<Bucket*> resolve_path_;

  // Contiguous record pools (deque = stable addresses for Bucket*/coord ptrs)
  std::deque<ARecord> a_pool_;
  std::vector<ARecord*> a_by_id_;
  std::size_t a_live_ = 0;
  std::deque<BRecord> b_pool_;
  std::vector<BRecord*> b_by_id_;
  std::vector<BRecord*> b_free_;
  std::size_t b_live_ = 0;
  std::vector<PointId> nn_B_;
  std::vector<double, AlignedAlloc<double>> B_flat_;  // 精确 NN 的连续 B 坐标，64 字节对齐
  // 唯一坐标池，float32。f32bin 的值经 double 再存回来位型不变；查询时再展开成 double。
  std::vector<float> A_flat_;
  mutable std::vector<double> coord_view_;  // coordsA 的一次性缓冲
  std::vector<double, AlignedAlloc<double>> nn_pack_;  // 查询时收拢抽中的 A 行
  std::unordered_map<PointId, size_t> b_nn_index_;

  PointId next_a_id_ = 1;
  PointId next_b_id_ = 1;

  void ensureDim(const std::vector<double>& x) const {
    if (static_cast<int>(x.size()) != dim_)
      throw std::invalid_argument("dimension mismatch");
  }
  void ensureCapacityA() {
    if (static_cast<int>(a_live_) + 1 > n_max_)
      throw std::runtime_error("|A| would exceed n_max");
  }
  void ensureCapacityB() {
    if (static_cast<int>(b_live_) + 1 > n_max_)
      throw std::runtime_error("|B| would exceed n_max");
  }

  ARecord* aRec(PointId id) const {
    if (id >= a_by_id_.size() || !a_by_id_[id])
      throw std::runtime_error("unknown A id");
    return a_by_id_[id];
  }
  BRecord* bRec(PointId id) const {
    if (id >= b_by_id_.size() || !b_by_id_[id])
      throw std::runtime_error("unknown B id");
    return b_by_id_[id];
  }
  BRecord* allocB() {
    if (!b_free_.empty()) {
      BRecord* rec = b_free_.back();
      b_free_.pop_back();
      *rec = BRecord{};
      return rec;
    }
    b_pool_.emplace_back();
    return &b_pool_.back();
  }
  void freeB(BRecord* rec) {
    *rec = BRecord{};
    b_free_.push_back(rec);
  }

  // Range and H_0 are already done. Attach this A point and refresh the frontier.
  PointId insertAStructure(const std::vector<double>& x, BucketKey& h0) {
    using clock = std::chrono::high_resolution_clock;

    PointId id = next_a_id_++;
    a_pool_.emplace_back();
    ARecord* rec = &a_pool_.back();
    rec->id = id;
    if (a_by_id_.size() <= id) a_by_id_.resize(id + 1, nullptr);
    a_by_id_[id] = rec;
    ++a_live_;

    // exact layer (coords written into A_flat_ at most once per unique point)
    auto t0 = clock::now();
    size_t coff = 0;
    ExactCell* cell = getOrCreateExact(x.data(), &coff);  // 精确叶子：同坐标共用一格
    rec->coord_off = coff;
    cell->A_count += 1;
    rec->exact_pos = cell->A_members.size();
    cell->A_members.push_back(id);
    rec->cell = cell;
    auto t1 = clock::now();
    prof_.insertA_exact_ms += msBetween(t0, t1);

    // E2LSH layers: create along new path; once an existing bucket is hit,
    // climb ancestors via parent pointers (no more parentKey/HashMap).
    t0 = clock::now();
    resolveLayerPath(h0);
    t1 = clock::now();
    prof_.insertA_resolve_ms += msBetween(t0, t1);
    const std::vector<Bucket*>& path = resolve_path_;
    for (int i = 0; i <= L_; ++i) {
      Bucket* buck = path[i];
      buck->A_count += 1;
      publishChildWeight(buck);
      // Member lists only at L0; higher layers sample via children tree.
      if (i == 0) buck->A_members.push_back(id);
    }
    // 分层：不用给 0..L 每一层都改采样器，只刷新「精确格」和路上那一个前沿桶。
    cell->l0_bucket = path[0];
    syncExactStratum(cell);
    syncPathFrontierBucket(path);

    // Drop full K-tuples on non-top buckets now that L0 members / parent links exist.
    chillPathKeys(path);

    ++prof_.insertA_count;
    return id;
  }

  PointId insertAMembers(const std::vector<double>& x, ExactCell* cell, size_t coff,
                         Bucket* const* path) {
    (void)x;
    PointId id = next_a_id_++;
    a_pool_.emplace_back();
    ARecord* rec = &a_pool_.back();
    rec->id = id;
    if (a_by_id_.size() <= id) a_by_id_.resize(id + 1, nullptr);
    a_by_id_[id] = rec;
    ++a_live_;
    rec->coord_off = coff;
    rec->cell = cell;
    rec->exact_pos = cell->A_members.size();
    cell->A_members.push_back(id);
    cell->A_count += 1;
    for (int i = 0; i <= L_; ++i) {
      Bucket* buck = path[i];
      buck->A_count += 1;
      publishChildWeight(buck);
      if (i == 0) buck->A_members.push_back(id);
    }
    cell->l0_bucket = path[0];
    chillPathKeys(path, L_ + 1);
    ++prof_.insertA_count;
    return id;
  }

  // The first point is the anchor. R_seen is the farthest distance from it seen so far.
  // Levels grow until Δ_L >= 2 R_seen. Insert and delete do not redraw g_j.
  void updateRangeAndExpand(const std::vector<double>& x) {
    if (!has_anchor_) {
      anchor_ = x;
      has_anchor_ = true;
      R_seen_ = 0.0;
    } else {
      R_seen_ = std::max(R_seen_, l2(x, anchor_));
    }
    while (deltas_[L_] < 2.0 * R_seen_ - 1e-15) {
      expandTop();
    }
  }

  void fillH0(const double* xd, BucketKey& key) const {
    e2lsh::fillH0(xd, dim_, K_, g_flat_.data(), S0_.data(), rho_, key);
  }
  BucketKey computeH0(const double* xd) const {
    BucketKey key;
    fillH0(xd, key);
    return key;
  }
  BucketKey computeH0(const std::vector<double>& x) const { return computeH0(x.data()); }

  void parentKeyInPlace(BucketKey& z, int i) const {
    // Parent id uses floor((h + ξ) / 2).
    e2lsh::parentKeyInPlace(z, K_, xi_[static_cast<size_t>(i)]);
  }

  BucketKey parentKey(const BucketKey& z, int i) const {
    BucketKey p = z;
    parentKeyInPlace(p, i);
    return p;
  }

  // Open one coarser level: double Δ, and give every old top bucket a parent.
  // The new layer's bits ξ are drawn from the range seen so far.
  void expandTop() {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    std::bernoulli_distribution bern(0.5);
    std::vector<uint8_t> bits(K_);
    for (int j = 0; j < K_; ++j) bits[j] = bern(rng_) ? 1 : 0;  // 每根投影抛一枚硬币
    xi_.push_back(bits);

    int oldL = L_;
    int newL = oldL + 1;
    deltas_.push_back(deltas_[oldL] * 2.0);  // Δ_{L+1} = 2 Δ_L
    layer_maps_.emplace_back();

    // EXPAND-TOP: aggregate from old top only. Top-layer keys are kept warm;
    // parent locate uses thin fingerprint map + exact K-tuple compare.
    forEachBucket(oldL, [&](Bucket* child) {
      ensureBucketKey(child);
      BucketKey ck = keyOfWarm(child);
      BucketKey pk = parentKey(ck, oldL);
      Bucket* parent = getOrCreateBucket(newL, pk);
      linkParent(child, parent);
      parent->A_count += child->A_count;
      publishChildWeight(parent);
      parent->B_count += child->B_count;
    });

    // Extend cached B membership via parent pointers (no re-hash / no history scan)
    for (PointId id = 1; id < b_by_id_.size(); ++id) {
      BRecord* brec = b_by_id_[id];
      if (!brec) continue;
      if (static_cast<int>(brec->membership.size()) != oldL + 1)
        throw std::runtime_error("expandTop: stale B membership");
      Bucket* child = brec->membership[oldL];
      if (!child->parent) throw std::runtime_error("expandTop: missing parent ptr");
      brec->membership.resize(newL + 1);
      brec->membership[newL] = child->parent;
    }

    L_ = newL;
    ensureStratumSize();
    // The old top may move from the fallback into the new real layer S_{newL}.
    forEachBucket(oldL, [&](Bucket* b) { syncBucketStratum(b); });
    forEachBucket(newL, [&](Bucket* b) { syncBucketStratum(b); });

    // Old top is no longer L_: drop full K-tuples (reconstructable via children).
    forEachBucket(oldL, [&](Bucket* b) { chillBucketKey(b); });

    auto t1 = clock::now();
    prof_.expandTop_ms += msBetween(t0, t1);
    ++prof_.expandTop_count;
  }

  // float→double，和插入时的 static_cast<double>(float) 同一条路。
  void loadARow(const ARecord* arec, double* dst) const {
    const float* s = A_flat_.data() + arec->coord_off;
    const int D = dim_;
    int i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= D; i += 8) {
      _mm512_storeu_pd(dst + i, _mm512_cvtps_pd(_mm256_loadu_ps(s + i)));
    }
#endif
    for (; i < D; ++i) dst[i] = static_cast<double>(s[i]);
  }

  size_t appendCoords(const double* xd) {
    const size_t D = static_cast<size_t>(dim_);
    const size_t off = A_flat_.size();
    A_flat_.resize(off + D);
    float* dst = A_flat_.data() + off;
    int i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= static_cast<int>(D); i += 8) {
      _mm256_storeu_ps(dst + i, _mm512_cvtpd_ps(_mm512_loadu_pd(xd + i)));
    }
#endif
    for (; i < static_cast<int>(D); ++i) dst[i] = static_cast<float>(xd[i]);
    return off;
  }

  // Exact-cell lookup compares the two fingerprints.
  ExactCell* getOrCreateExactHashed(const double* xd, std::size_t h1, std::size_t h2,
                                    size_t* out_off = nullptr) {
    const uint32_t id = exact_map_.find(h1, h2, [&](uint32_t e) { return exact_by_id_[e]->coord_hash2; });
    if (id) {
      ExactCell* cell = exact_by_id_[id];
      if (out_off) *out_off = cell->coord_off;
      return cell;
    }
    const size_t off = appendCoords(xd);
    exact_pool_.emplace_back();
    ExactCell* ptr = &exact_pool_.back();
    ptr->id = static_cast<uint32_t>(exact_by_id_.size());
    ptr->coord_off = off;
    ptr->coord_hash = h1;
    ptr->coord_hash2 = h2;
    exact_by_id_.push_back(ptr);
    exact_map_.insert(h1, ptr->id);
    if (out_off) *out_off = off;
    return ptr;
  }

  ExactCell* getOrCreateExact(const double* xd, size_t* out_off = nullptr) {
    std::size_t h1 = 0, h2 = 0;
    hashCoords(xd, dim_, h1, h2);
    return getOrCreateExactHashed(xd, h1, h2, out_off);
  }
  ExactCell* getOrCreateExact(const std::vector<double>& x, size_t* out_off = nullptr) {
    return getOrCreateExact(x.data(), out_off);
  }

  ExactCell* findExact(const double* xd) {
    std::size_t h1 = 0, h2 = 0;
    hashCoords(xd, dim_, h1, h2);
    const uint32_t id = exact_map_.find(h1, h2, [&](uint32_t e) { return exact_by_id_[e]->coord_hash2; });
    return id ? exact_by_id_[id] : nullptr;
  }
  ExactCell* findExact(const std::vector<double>& x) { return findExact(x.data()); }


  // 从 H_0 走到 H_L：还没见过的层就建桶，一旦撞上已有桶，后面全跟 parent 指针爬。
  // 结果写在 resolve_path_，下次调用会被覆盖。
  void resolveLayerPath(BucketKey& cur) {
    resolve_path_.assign(static_cast<size_t>(L_) + 1, nullptr);
    Bucket* prev = nullptr;
    bool via_parent = false;
    for (int i = 0; i <= L_; ++i) {
      Bucket* buck = nullptr;
      if (via_parent) {
        if (!prev || !prev->parent)
          throw std::runtime_error("resolveLayerPath: missing parent pointer");
        buck = prev->parent;
      } else {
        buck = findBucket(i, cur);
        if (buck) {
          via_parent = true;
        } else {
          buck = createBucket(i, cur);
        }
      }
      if (prev) linkParent(prev, buck);
      resolve_path_[static_cast<size_t>(i)] = buck;
      prev = buck;
      if (!via_parent && i < L_) parentKeyInPlace(cur, i);
    }
  }

  template <class Fn>
  void forEachBucket(int layer, Fn&& fn) {
    for (uint32_t id = 1; id < buckets_.size(); ++id) {
      Bucket* b = buckets_[id];
      if (b && b->layer == layer) fn(b);
    }
  }

  static BucketKey keyOfWarm(const Bucket* b) {
    BucketKey k;
    k.k = static_cast<int>(b->key_h.size());
    k.h = b->key_h;
    k.hash = b->key_hash;
    k.hash2 = b->key_hash2;
    return k;
  }

  bool canReconstructKey(const Bucket* b) const {
    if (!b) return false;
    if (b->layer == 0) return !b->A_members.empty();
    return !b->children.empty();
  }

  void chillBucketKey(Bucket* b) {
    // The full K-tuple stays only on the coarsest layer. A colder bucket that can
    // be rebuilt from its children or members drops key_h.
    if (!b) return;
    if (b->layer >= L_) return;
    if (!canReconstructKey(b)) return;
    if (b->key_h.empty()) return;
    b->key_h.clear();
    b->key_h.shrink_to_fit();
  }

  void chillPathKeys(Bucket* const* path, int n) {
    for (int i = 0; i < n; ++i) chillBucketKey(path[i]);
  }
  void chillPathKeys(const std::vector<Bucket*>& path) {
    if (!path.empty()) chillPathKeys(path.data(), static_cast<int>(path.size()));
  }

  void ensureBucketKey(Bucket* b) {
    if (!b) throw std::runtime_error("ensureBucketKey: null");
    if (static_cast<int>(b->key_h.size()) == K_) return;
    if (b->layer == 0) {
      if (b->A_members.empty())
        throw std::runtime_error("ensureBucketKey: L0 has no A_members to reconstruct");
      ARecord* arec = aRec(b->A_members[0]);
      std::vector<double> row(static_cast<std::size_t>(dim_));
      loadARow(arec, row.data());
      BucketKey k = computeH0(row.data());
      if (k.hash != b->key_hash)
        throw std::runtime_error("ensureBucketKey: L0 hash mismatch on reconstruct");
      b->key_h = std::move(k.h);
      return;
    }
    if (b->children.empty())
      throw std::runtime_error("ensureBucketKey: non-L0 has no children to reconstruct");
    Bucket* child = b->children[0];
    ensureBucketKey(child);
    BucketKey ck = keyOfWarm(child);
    BucketKey pk = parentKey(ck, b->layer - 1);
    if (pk.hash != b->key_hash)
      throw std::runtime_error("ensureBucketKey: parent hash mismatch on reconstruct");
    b->key_h = std::move(pk.h);
  }

  bool bucketKeyEquals(Bucket* b, const BucketKey& key) {
    if (b->key_hash != key.hash) return false;
    if (static_cast<int>(b->key_h.size()) != K_) ensureBucketKey(b);
    if (key.k != K_) return false;
    return std::memcmp(b->key_h.data(), key.h.data(),
                       static_cast<size_t>(K_) * sizeof(int64_t)) == 0;
  }

  Bucket* allocBucket() {
    if (!bucket_free_.empty()) {
      Bucket* p = bucket_free_.back();
      const uint32_t id = p->id;
      bucket_free_.pop_back();
      *p = Bucket{};
      p->id = id;
      buckets_[id] = p;
      return p;
    }
    bucket_pool_.emplace_back();
    Bucket* p = &bucket_pool_.back();
    p->id = static_cast<uint32_t>(buckets_.size());
    buckets_.push_back(p);
    return p;
  }

  uint32_t allocFilledBucket(int layer, uint64_t h1, uint64_t h2, const int64_t* h) {
    std::lock_guard<std::mutex> lock(bucket_alloc_mu_);
    Bucket* ptr = allocBucket();
    ptr->layer = layer;
    ptr->key_hash = static_cast<std::size_t>(h1);
    ptr->key_hash2 = static_cast<std::size_t>(h2);
    if (layer >= L_) ptr->key_h.assign(h, h + K_);
    ptr->children.reserve(4);
    ptr->child_weight.reserve(4);
    return ptr->id;
  }

  Bucket* createBucket(int layer, const BucketKey& key) {
    Bucket* ptr = allocBucket();
    ptr->layer = layer;
    ptr->key_hash = key.hash;
    ptr->key_hash2 = key.hash2;
    if (layer >= L_) ptr->key_h = key.h;  // full K-tuple only on the current coarsest layer
    ptr->children.reserve(4);
    ptr->child_weight.reserve(4);
    layer_maps_[static_cast<size_t>(layer)].insert(key.hash, ptr->id);
    return ptr;
  }

  bool tryFillParentChain(int from_layer, Bucket* node, Bucket** path) const {
    if (from_layer >= L_) return true;
    Bucket* x = node;
    for (int j = from_layer + 1; j <= L_; ++j) {
      x = x->parent;
      if (!x) return false;
      path[j] = x;
    }
    return true;
  }

  void internLayerPathLocked(int64_t* h_row, uint64_t h1, uint64_t h2, Bucket** path) {
    Bucket* prev = nullptr;
    for (int i = 0; i <= L_; ++i) {
      const uint32_t id = layer_maps_[static_cast<size_t>(i)].findOrCreate(
          h1, h2, [&](uint32_t bid) { return buckets_[bid]->key_hash2; },
          [&, i, h1, h2]() { return allocFilledBucket(i, h1, h2, h_row); });
      Bucket* buck = buckets_[id];
      if (prev) prev->parent = buck;
      path[i] = buck;
      prev = buck;
      if (i < L_) {
        if (tryFillParentChain(i, buck, path)) return;
        parentKeyRowInPlace(h_row, K_, xi_[static_cast<size_t>(i)], h1, h2);
      }
    }
  }

  void rebuildRamChildLists() {
    for (uint32_t id = 1; id < buckets_.size(); ++id) {
      Bucket* b = buckets_[id];
      if (!b) continue;
      b->children.clear();
      b->child_weight.clear();
      b->child_pos = -1;
    }
    for (uint32_t id = 1; id < buckets_.size(); ++id) {
      Bucket* b = buckets_[id];
      if (!b || !b->parent) continue;
      Bucket* p = b->parent;
      b->child_pos = static_cast<int>(p->children.size());
      p->children.push_back(b);
      p->child_weight.push_back(b->A_count);
    }
  }

  void rebuildStrata() {
    ensureStratumSize();
    const int nS = L_ + 2;
    stratum_samp_.assign(static_cast<size_t>(nS), DynamicWeightedSampler{});
    stratum_N_.assign(static_cast<size_t>(nS), 0);
    stratum_active_.assign(static_cast<size_t>(nS), {});
    for (uint32_t i = 1; i < exact_by_id_.size(); ++i) {
      ExactCell* c = exact_by_id_[i];
      if (!c) continue;
      c->in_stratum = false;
      c->stratum_idx = -1;
      c->stratum_dense_pos = -1;
      c->stratum_slot = SamplerSlotRef{};
      setExactStratum(c, desiredExactStratum(c));
    }
    for (uint32_t id = 1; id < buckets_.size(); ++id) {
      Bucket* b = buckets_[id];
      if (!b) continue;
      b->in_stratum = false;
      b->stratum_idx = -1;
      b->stratum_dense_pos = -1;
      b->stratum_slot = SamplerSlotRef{};
      setBucketStratum(b, desiredBucketStratum(b));
    }
  }

  Bucket* getOrCreateBucket(int layer, const BucketKey& key) {
    Bucket* existing = findBucket(layer, key);
    if (existing) return existing;
    return createBucket(layer, key);
  }

  // Layer lookup compares fingerprints, not the full K-tuple.
  Bucket* findBucket(int layer, const BucketKey& key) {
    const uint32_t id = layer_maps_[static_cast<size_t>(layer)].find(
        key.hash, key.hash2, [&](uint32_t bid) { return buckets_[bid]->key_hash2; });
    return id ? buckets_[id] : nullptr;
  }

  void eraseBucketFromLayer(int layer, Bucket* b) {
    if (!b || !b->id) return;
    layer_maps_[static_cast<size_t>(layer)].erase(b->key_hash, b->id);
    buckets_[b->id] = b;
    bucket_free_.push_back(b);
  }

  // Drop the cell from the dictionary once both A and B are gone.
  void maybeCleanupExact(ExactCell* cell) {
    if (cell->A_count == 0 && cell->B_count == 0) {
      clearExactStratum(cell);
      exact_map_.erase(cell->coord_hash, cell->id);
    } else if (cell->A_count == 0 && cell->B_count > 0) {
      clearExactStratum(cell);
    }
  }

  void maybeCleanupBucket(int layer, Bucket* b) {
    if (b->A_count == 0 && b->B_count == 0) {
      clearBucketStratum(b);
      unlinkFromParent(b);
      // Detach children (should already be empty of A/B, but keep tree consistent)
      for (Bucket* child : b->children) {
        child->parent = nullptr;
        child->child_pos = -1;
      }
      b->children.clear();
      b->child_weight.clear();
      eraseBucketFromLayer(layer, b);
    } else if (b->A_count == 0 && b->B_count > 0) {
      clearBucketStratum(b);
    }
  }

  // Swap-delete at known index. Returns PointId swapped into slot (0 if removed last).
  // Caller must update the moved point's stored pos to `pos`.
  static PointId removeFromMemberListAt(std::vector<PointId>& members, size_t pos) {
    if (pos >= members.size()) throw std::runtime_error("member pos OOB");
    size_t last = members.size() - 1;
    PointId moved = 0;
    if (pos != last) {
      moved = members[last];
      members[pos] = moved;
    }
    members.pop_back();
    return moved;
  }

  // Uniform draw inside a bucket whose B is empty. Children are scanned by A_count,
  // so one descent is linear in the number of children. The draw stays uniform.
  PointId sampleAFromBucket(Bucket* b) {
    if (!b || b->A_count <= 0) throw std::runtime_error("empty bucket A_count");
    if (!b->children.empty()) {
      int r = std::uniform_int_distribution<int>(1, b->A_count)(rng_);  // 在 1..A_v 里抽一个座位号
      const int n = static_cast<int>(b->child_weight.size());
      const int* w = b->child_weight.data();
      int acc = 0;
      for (int i = 0; i < n; ++i) {
        acc += w[i];
        if (r <= acc) return sampleAFromBucket(b->children[static_cast<size_t>(i)]);
      }
      return sampleAFromBucket(b->children.back());
    }
    if (b->A_members.empty()) throw std::runtime_error("empty L0 members");
    std::uniform_int_distribution<size_t> dist(0, b->A_members.size() - 1);
    return b->A_members[dist(rng_)];  // 到了第 0 层，点名册上均匀抽
  }

  PointId sampleAFromComponent(const ComponentId& cid) {
    if (cid.kind == ComponentKind::Exact) {
      const auto& members = static_cast<ExactCell*>(cid.ptr)->A_members;
      if (members.empty()) throw std::runtime_error("empty component members");
      std::uniform_int_distribution<size_t> dist(0, members.size() - 1);
      return members[dist(rng_)];
    }
    return sampleAFromBucket(static_cast<Bucket*>(cid.ptr));
  }

  // 嵌套保证：B 非空的层一定是后缀——细的撞上了，粗的不可能再空。
  // So the search can be binary: j = min{i : bucket B_count > 0}, or L+1 if none.
  int firstCollisionLayer(const ARecord* arec, int lo_min = 0) const {
    std::vector<Bucket*> path(static_cast<size_t>(L_) + 1, nullptr);
    Bucket* b = arec->cell->l0_bucket;
    for (int i = 0; i <= L_; ++i) {
      if (!b) throw std::runtime_error("firstCollisionLayer: broken parent chain");
      path[static_cast<size_t>(i)] = b;
      b = b->parent;
    }
    int lo = lo_min;
    int hi = L_ + 1;  // sentinel: no real-layer collision
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (mid > L_) {
        hi = mid;
        continue;
      }
      if (path[static_cast<size_t>(mid)]->B_count > 0) hi = mid;
      else lo = mid + 1;
    }
    return lo;
  }

  // D_a is Δ_j at the first colliding layer, and 2 Δ_L when no real layer collides.
  double daFromFirstCollision(int j) const {
    if (j <= L_) return deltas_[static_cast<size_t>(j)];
    return 2.0 * deltas_[static_cast<size_t>(L_)];
  }

  double computeDaConst(const ARecord* arec) const {
    if (arec->cell->B_count > 0) return 0.0;
    return daFromFirstCollision(firstCollisionLayer(arec, 0));
  }

  // Exact NN scans B_flat_. The scan is O(|B| d).
  double exactNN(const double* x) const {
    return simd_l2::min_l2_flat(x, dim_, B_flat_.data(), nn_B_.size());
  }

  double exactNNOfA(const ARecord* arec) const {
    std::vector<double> row(static_cast<std::size_t>(dim_));
    loadARow(arec, row.data());
    return exactNN(row.data());
  }

  // 查询时把抽中的 float 行展开成一小块 double，再对窗口 B 做距离。
  // f32→f64 不改变位型，距离核仍是 min_l2_flat。缓冲相对 B 错开 128 字节。
  template <class Task>
  void evalQueryNN(const std::vector<Task>& tasks, std::vector<double>& da_out) {
    const int ntasks = static_cast<int>(tasks.size());
    da_out.assign(static_cast<std::size_t>(ntasks), 0.0);
    if (ntasks == 0 || nn_B_.empty()) return;
    const std::size_t D = static_cast<std::size_t>(dim_);
    nn_pack_.resize(static_cast<std::size_t>(ntasks) * D + 64);
    std::uintptr_t row0 = reinterpret_cast<std::uintptr_t>(nn_pack_.data());
    row0 = (row0 + 63u) & ~std::uintptr_t{63};
    const std::uintptr_t bbase = reinterpret_cast<std::uintptr_t>(B_flat_.data());
    if (((row0 - bbase) & 511u) == 0) row0 += 128;
    double* base = reinterpret_cast<double*>(row0);
    for (int i = 0; i < ntasks; ++i) {
      loadARow(aRec(tasks[static_cast<std::size_t>(i)].aid), base + static_cast<std::size_t>(i) * D);
    }
    const double* B = B_flat_.data();
    const std::size_t nB = nn_B_.size();
    int threads = 1;
#if defined(_OPENMP)
    threads = omp_get_max_threads();
#endif
    if (threads > 1 && ntasks >= 64) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (int i = 0; i < ntasks; ++i) {
        da_out[static_cast<std::size_t>(i)] =
            simd_l2::min_l2_flat(base + static_cast<std::size_t>(i) * D, dim_, B, nB);
      }
    } else {
      for (int i = 0; i < ntasks; ++i) {
        da_out[static_cast<std::size_t>(i)] =
            simd_l2::min_l2_flat(base + static_cast<std::size_t>(i) * D, dim_, B, nB);
      }
    }
  }
  double exactNN(const std::vector<double>& x) const { return exactNN(x.data()); }

  // Frontier maintenance: one Fenwick sampler per layer.

  void ensureStratumSize() {
    const int need = L_ + 2;
    if (static_cast<int>(stratum_samp_.size()) < need) {
      stratum_samp_.resize(static_cast<size_t>(need));
      stratum_N_.resize(static_cast<size_t>(need), 0);
      stratum_active_.resize(static_cast<size_t>(need));
    }
  }
  void ensureStratumSizeConst() const {
    const int need = L_ + 2;
    if (static_cast<int>(stratum_samp_.size()) < need) {
      stratum_samp_.resize(static_cast<size_t>(need));
      stratum_N_.resize(static_cast<size_t>(need), 0);
      stratum_active_.resize(static_cast<size_t>(need));
    }
  }

  void linkParent(Bucket* child, Bucket* parent) {
    if (!child) return;
    if (child->parent == parent) return;
    unlinkFromParent(child);
    child->parent = parent;
    if (parent) {
      child->child_pos = static_cast<int>(parent->children.size());
      parent->children.push_back(child);
      parent->child_weight.push_back(child->A_count);
    }
  }

  static void publishChildWeight(Bucket* b) {
    if (!b || !b->parent || b->child_pos < 0) return;
    Bucket* p = b->parent;
    const int pos = b->child_pos;
    if (pos < static_cast<int>(p->child_weight.size()))
      p->child_weight[static_cast<size_t>(pos)] = b->A_count;
  }

  void unlinkFromParent(Bucket* child) {
    if (!child || !child->parent || child->child_pos < 0) {
      if (child) {
        child->parent = nullptr;
        child->child_pos = -1;
      }
      return;
    }
    Bucket* parent = child->parent;
    int pos = child->child_pos;
    auto& ch = parent->children;
    auto& wt = parent->child_weight;
    if (pos >= 0 && pos < static_cast<int>(ch.size()) && ch[static_cast<size_t>(pos)] == child) {
      int last = static_cast<int>(ch.size()) - 1;
      if (static_cast<int>(wt.size()) == static_cast<int>(ch.size())) {
        if (pos != last) wt[static_cast<size_t>(pos)] = wt[static_cast<size_t>(last)];
        wt.pop_back();
      }
      if (pos != last) {
        Bucket* moved = ch[static_cast<size_t>(last)];
        ch[static_cast<size_t>(pos)] = moved;
        moved->child_pos = pos;
      }
      ch.pop_back();
    }
    child->parent = nullptr;
    child->child_pos = -1;
  }

  // Parent occupied and own B empty: the subtree's A points belong to the parent's stratum.
  // A coarsest bucket with empty B goes to the fallback S_{L+1}.
  int desiredBucketStratum(const Bucket* b) const {
    if (!b || b->A_count <= 0 || b->B_count != 0) return -1;
    if (b->layer == L_) return L_ + 1;  // coarsest empty-B bucket goes to the fallback
    if (b->parent && b->parent->B_count > 0) return b->layer + 1;  // 首碰发生在父亲那一层
    return -1;
  }

  // 精确叶子版：格子 B 空、L0 桶 B 非空 → 整格属于 S_0。
  int desiredExactStratum(const ExactCell* cell) const {
    if (!cell || cell->A_count <= 0 || cell->B_count != 0) return -1;
    if (cell->l0_bucket && cell->l0_bucket->B_count > 0) return 0;
    return -1;
  }

  void stratumActiveRemove(int idx, int pos, ComponentKind kind) {
    if (idx < 0 || idx >= static_cast<int>(stratum_active_.size())) return;
    auto& v = stratum_active_[static_cast<size_t>(idx)];
    if (pos < 0 || pos >= static_cast<int>(v.size())) return;
    int last = static_cast<int>(v.size()) - 1;
    if (pos != last) {
      ComponentId moved = v[static_cast<size_t>(last)];
      v[static_cast<size_t>(pos)] = moved;
      if (moved.kind == ComponentKind::Exact) {
        static_cast<ExactCell*>(moved.ptr)->stratum_dense_pos = pos;
      } else {
        static_cast<Bucket*>(moved.ptr)->stratum_dense_pos = pos;
      }
    }
    v.pop_back();
    (void)kind;
  }

  void clearBucketStratum(Bucket* b) {
    if (!b || !b->in_stratum) return;
    ensureStratumSize();
    int idx = b->stratum_idx;
    if (idx >= 0 && idx < static_cast<int>(stratum_samp_.size())) {
      int w = static_cast<int>(std::lround(stratum_samp_[static_cast<size_t>(idx)].weightAt(b->stratum_slot)));
      stratum_N_[static_cast<size_t>(idx)] -= w;
      if (stratum_N_[static_cast<size_t>(idx)] < 0) stratum_N_[static_cast<size_t>(idx)] = 0;
      stratum_samp_[static_cast<size_t>(idx)].destroy(b->stratum_slot);
      stratumActiveRemove(idx, b->stratum_dense_pos, ComponentKind::Bucket);
    }
    b->in_stratum = false;
    b->stratum_idx = -1;
    b->stratum_dense_pos = -1;
    b->stratum_slot = SamplerSlotRef{};
  }

  void clearExactStratum(ExactCell* cell) {
    if (!cell || !cell->in_stratum) return;
    ensureStratumSize();
    int idx = cell->stratum_idx;
    if (idx >= 0 && idx < static_cast<int>(stratum_samp_.size())) {
      int w = static_cast<int>(std::lround(stratum_samp_[static_cast<size_t>(idx)].weightAt(cell->stratum_slot)));
      stratum_N_[static_cast<size_t>(idx)] -= w;
      if (stratum_N_[static_cast<size_t>(idx)] < 0) stratum_N_[static_cast<size_t>(idx)] = 0;
      stratum_samp_[static_cast<size_t>(idx)].destroy(cell->stratum_slot);
      stratumActiveRemove(idx, cell->stratum_dense_pos, ComponentKind::Exact);
    }
    cell->in_stratum = false;
    cell->stratum_idx = -1;
    cell->stratum_dense_pos = -1;
    cell->stratum_slot = SamplerSlotRef{};
  }

  // Place or remove a bucket in one layer's Fenwick. The weight is A_count.
  void setBucketStratum(Bucket* b, int idx) {
    ensureStratumSize();
    if (!b || idx < 0) {
      clearBucketStratum(b);
      return;
    }
    const double w = static_cast<double>(b->A_count);
    if (b->in_stratum && b->stratum_idx == idx) {
      double oldw = stratum_samp_[static_cast<size_t>(idx)].weightAt(b->stratum_slot);
      stratum_samp_[static_cast<size_t>(idx)].setWeight(b->stratum_slot, w);
      stratum_N_[static_cast<size_t>(idx)] += static_cast<int>(std::lround(w - oldw));
      return;
    }
    clearBucketStratum(b);
    ComponentId id;
    id.kind = ComponentKind::Bucket;
    id.layer = b->layer;
    id.ptr = b;
    b->stratum_slot = stratum_samp_[static_cast<size_t>(idx)].create(w, id);
    b->in_stratum = true;
    b->stratum_idx = idx;
    b->stratum_dense_pos = static_cast<int>(stratum_active_[static_cast<size_t>(idx)].size());
    stratum_active_[static_cast<size_t>(idx)].push_back(id);
    stratum_N_[static_cast<size_t>(idx)] += b->A_count;
  }

  void setExactStratum(ExactCell* cell, int idx) {
    ensureStratumSize();
    if (!cell || idx < 0) {
      clearExactStratum(cell);
      return;
    }
    const double w = static_cast<double>(cell->A_count);
    if (cell->in_stratum && cell->stratum_idx == idx) {
      double oldw = stratum_samp_[static_cast<size_t>(idx)].weightAt(cell->stratum_slot);
      stratum_samp_[static_cast<size_t>(idx)].setWeight(cell->stratum_slot, w);
      stratum_N_[static_cast<size_t>(idx)] += static_cast<int>(std::lround(w - oldw));
      return;
    }
    clearExactStratum(cell);
    ComponentId id;
    id.kind = ComponentKind::Exact;
    id.ptr = cell;
    cell->stratum_slot = stratum_samp_[static_cast<size_t>(idx)].create(w, id);
    cell->in_stratum = true;
    cell->stratum_idx = idx;
    cell->stratum_dense_pos = static_cast<int>(stratum_active_[static_cast<size_t>(idx)].size());
    stratum_active_[static_cast<size_t>(idx)].push_back(id);
    stratum_N_[static_cast<size_t>(idx)] += cell->A_count;
  }

  void syncBucketStratum(Bucket* b) {
    if (!b) return;
    setBucketStratum(b, desiredBucketStratum(b));
  }

  void syncExactStratum(ExactCell* cell) {
    if (!cell) return;
    setExactStratum(cell, desiredExactStratum(cell));
  }

  // A 点更新时 B 拓扑没变：整条路上只有「第一个有 B 的桶」下面那个孩子可能改层籍。
  void syncPathFrontierBucket(const std::vector<Bucket*>& path) {
    if (path.empty()) return;
    int j = L_ + 1;
    for (int i = 0; i <= L_; ++i) {
      if (path[static_cast<size_t>(i)]->B_count > 0) {
        j = i;
        break;
      }
    }
    if (j == 0) return;  // collision at layer 0 => no B-empty frontier bucket
    if (j <= L_) syncBucketStratum(path[static_cast<size_t>(j - 1)]);
    else syncBucketStratum(path[static_cast<size_t>(L_)]);
  }

  void syncExactCellsTouchingL0(Bucket* l0) {
    if (!l0 || l0->layer != 0) return;
    // Dedup ExactCell* via generation stamp (comment previously claimed a set but did not).
    const uint64_t g = ++sync_gen_;
    if (sync_gen_ == 0) {  // wrap
      sync_gen_ = 1;
      for (uint32_t i = 1; i < exact_by_id_.size(); ++i)
        if (exact_by_id_[i]) exact_by_id_[i]->sync_stamp = 0;
    }
    for (PointId aid : l0->A_members) {
      ExactCell* cell = aRec(aid)->cell;
      if (cell->sync_stamp == g) continue;
      cell->sync_stamp = g;
      syncExactStratum(cell);
    }
  }

  // The frontier is stored on the children, so every child that holds A is synced.
  // Layer 0 also scans the exact cells in the bucket.
  void onBucketBTransition(Bucket* b) {
    if (!b) return;
    syncBucketStratum(b);
    for (Bucket* child : b->children) {
      if (child->A_count > 0) syncBucketStratum(child);
    }
    if (b->layer == 0) syncExactCellsTouchingL0(b);
  }

  // First-collision weights. scale[i] = Δ_i, fallback scale[L+1] = 2 Δ_L, W_i = N_i * scale[i].
  void stratumImportanceWeights(std::vector<double>& W, std::vector<double>& scale, double& D) {
    const int nS = L_ + 2;
    ensureStratumSize();
    W.assign(static_cast<size_t>(nS), 0.0);
    scale.assign(static_cast<size_t>(nS), 0.0);
    for (int i = 0; i <= L_; ++i) {
      scale[static_cast<size_t>(i)] = deltas_[static_cast<size_t>(i)];
      W[static_cast<size_t>(i)] =
          static_cast<double>(stratum_N_[static_cast<size_t>(i)]) * scale[static_cast<size_t>(i)];
    }
    scale[static_cast<size_t>(L_ + 1)] = 2.0 * deltas_[static_cast<size_t>(L_)];
    W[static_cast<size_t>(L_ + 1)] =
        static_cast<double>(stratum_N_[static_cast<size_t>(L_ + 1)]) * scale[static_cast<size_t>(L_ + 1)];
    D = 0.0;
    for (double w : W) D += w;
  }

  // Within a layer: draw a frontier component with probability proportional to A_u,
  // then draw uniformly inside it, so Pr[a | i] = 1/N_i.
  PointId sampleAFromStratum(int si) {
    ensureStratumSize();
    if (si < 0 || si >= static_cast<int>(stratum_samp_.size()))
      throw std::runtime_error("sampleAFromStratum: bad stratum");
    if (!(stratum_samp_[static_cast<size_t>(si)].totalWeight() > 0.0))
      throw std::runtime_error("sampleAFromStratum: empty stratum");
    ComponentId cid = stratum_samp_[static_cast<size_t>(si)].sample(rng_);
    return sampleAFromComponent(cid);
  }

  void collectAIdsUnderBucket(Bucket* b, std::vector<PointId>& out) const {
    if (!b) return;
    if (!b->children.empty()) {
      for (Bucket* ch : b->children) collectAIdsUnderBucket(ch, out);
      return;
    }
    out.insert(out.end(), b->A_members.begin(), b->A_members.end());
  }

  // Enumerate each point of the stratum once. Order follows the component list.
  template <typename TaskVec>
  void collectStratumAIds(int si, TaskVec& tasks) {
    ensureStratumSize();
    if (si < 0 || si >= static_cast<int>(stratum_active_.size())) return;
    for (const ComponentId& cid : stratum_active_[static_cast<size_t>(si)]) {
      if (cid.kind == ComponentKind::Exact) {
        for (PointId aid : static_cast<ExactCell*>(cid.ptr)->A_members) {
          tasks.push_back(typename TaskVec::value_type{aid, 1});
        }
      } else {
        std::vector<PointId> ids;
        collectAIdsUnderBucket(static_cast<Bucket*>(cid.ptr), ids);
        for (PointId aid : ids) tasks.push_back(typename TaskVec::value_type{aid, 1});
      }
    }
  }

  double sumDistancesInStratum(int si) {
    ensureStratumSize();
    double sum = 0.0;
    if (si < 0 || si >= static_cast<int>(stratum_active_.size())) return sum;
    const auto& active = stratum_active_[static_cast<size_t>(si)];
    // Collect aids then optional OpenMP over NN
    std::vector<PointId> aids;
    aids.reserve(static_cast<size_t>(stratum_N_[static_cast<size_t>(si)]));
    for (const ComponentId& cid : active) {
      if (cid.kind == ComponentKind::Exact) {
        for (PointId aid : static_cast<ExactCell*>(cid.ptr)->A_members) aids.push_back(aid);
      } else {
        collectAIdsUnderBucket(static_cast<Bucket*>(cid.ptr), aids);
      }
    }
    const int n = static_cast<int>(aids.size());
    std::vector<double> da(static_cast<size_t>(n), 0.0);
#if defined(_OPENMP)
    if (n >= 64) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i) {
        da[static_cast<size_t>(i)] = exactNNOfA(aRec(aids[static_cast<size_t>(i)]));
      }
    } else
#endif
    {
      for (int i = 0; i < n; ++i) {
        da[static_cast<size_t>(i)] = exactNNOfA(aRec(aids[static_cast<size_t>(i)]));
      }
    }
    for (double v : da) sum += v;
    return sum;
  }
};

}  // namespace e2lsh
