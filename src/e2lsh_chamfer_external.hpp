#pragma once
// External ℓ2 E2LSH Chamfer estimator. The directory stays in memory.
// Coordinates of A and bucket postings live on disk.
// Same algorithm as the in-memory estimator. Layer budgets call
// hamiltonAllocateGuaranteed. queryImportance is the ablation estimator;
// its coordinates are gathered from disk.
// Extra pieces here are finalizeA, coordinate gather, and posting gather.

#include "e2lsh_chamfer.hpp"
#include "os/fs.hpp"
#include "storage/coord_store.hpp"
#include "storage/posting_store.hpp"

#include <atomic>
#include <cstring>
#include <mutex>

namespace e2lsh {

class ExternalChamferEstimator {
 public:
  ExternalChamferEstimator(int dim, double rho, int n_max, uint64_t seed, const std::string& storage_dir,
                           int k_override = -1)
      : dim_(dim),
        rho_(rho),
        n_max_(n_max),
        K_(k_override > 0 ? k_override : computeK(n_max)),
        rng_(seed),
        storage_dir_(storage_dir) {
    if (!(rho_ > 0.0)) throw std::invalid_argument("rho must be > 0");
    if (dim_ <= 0) throw std::invalid_argument("dim must be > 0");
    if (K_ <= 0) throw std::invalid_argument("K must be > 0");
    if (storage_dir_.empty()) throw std::invalid_argument("storage_dir required");
    e2lsh::os::createDir(storage_dir_);

    std::normal_distribution<double> normal(0.0, 1.0);
    g_flat_.assign(static_cast<size_t>(K_) * static_cast<size_t>(dim_), 0.0);
    for (int j = 0; j < K_; ++j)
      for (int i = 0; i < dim_; ++i) g_flat_[static_cast<size_t>(j) * dim_ + i] = normal(rng_);
    double w0 = static_cast<double>(K_) * rho_;
    std::uniform_real_distribution<double> uni(0.0, w0);
    S0_.resize(K_);
    for (int j = 0; j < K_; ++j) S0_[j] = uni(rng_);
    L_ = 0;
    deltas_.push_back(rho_);
    xi_.clear();
    has_anchor_ = false;
    R_seen_ = 0.0;
    stratum_samp_.assign(2, DynamicWeightedSampler{});
    stratum_N_.assign(2, 0);
    stratum_active_.assign(2, {});

    recs_.resize(1);
    recs_live_ = 1;
    maps_.resize(1);
    layer_ids_.resize(1);
    exact_.resize(1);
    ex_off_.assign(1, 0);
    ex_len_.assign(1, 0);
    ex_cap_.assign(1, 0);
    a_by_id_.resize(1);
    b_by_id_.resize(1);

    coords_.open(storage_dir_ + "/coords.bin", dim_);
    postings_.open(storage_dir_ + "/postings.bin");
  }

  int K() const { return K_; }
  int L() const { return L_; }
  double rho() const { return rho_; }
  std::size_t sizeA() const { return a_live_; }
  std::size_t sizeB() const { return b_live_; }
  uint32_t nBuckets() const { return recs_live_ > 1 ? recs_live_ - 1 : 0; }
  bool storageDirect() const { return coords_.file().isDirect(); }
  bool storageIoUring() const { return coords_.file().hasIoUring(); }
  int64_t ioReads() const { return prof_.io_reads; }

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

  static int computeDefaultT(std::size_t n) { return defaultT(n); }

  void expandToCover(const std::vector<double>& x) {
    ensureDim(x);
    updateRangeAndExpand(x);
  }
  void expandToCoverR(double R) {
    if (!has_anchor_) throw std::runtime_error("expandToCoverR: no anchor");
    R_seen_ = std::max(R_seen_, R);
    while (deltas_[L_] < 2.0 * R_seen_ - 1e-15) expandTop();
    // Range is global; later insertA must not keep K-tuples on the live top
    // (GIST-1M would otherwise pin ~1 GiB of warm_keys_).
    keep_warm_keys_ = false;
    warm_keys_.clear();
  }

  void finalizeA() {
    if (!postings_.finalized()) {
      rebuildChildListsFromParents();
      postings_.finalize();
      coords_.fadviseDontNeed();
      snapIoStats();
    }
  }

  PointId insertA(const std::vector<double>& x) {
    using clock = std::chrono::high_resolution_clock;
    ensureDim(x);
    ensureCapacityA();
    double e0 = prof_.expandTop_ms;
    auto t0 = clock::now();
    updateRangeAndExpand(x);
    auto t1 = clock::now();
    prof_.insertA_other_ms += msBetween(t0, t1) - (prof_.expandTop_ms - e0);
    t0 = clock::now();
    fillH0(x.data(), h0_scratch_);
    t1 = clock::now();
    prof_.insertA_h0_ms += msBetween(t0, t1);
    return insertAStructure(x, h0_scratch_);
  }

  std::vector<PointId> insertABatch(const std::vector<std::vector<double>>& pts) {
    using clock = std::chrono::high_resolution_clock;
    if (pts.empty()) return {};
    const int n = static_cast<int>(pts.size());
    for (const auto& p : pts) ensureDim(p);
    if (static_cast<int>(a_live_) + n > n_max_) throw std::runtime_error("|A| would exceed n_max");

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
      for (int i = 0; i < n; ++i) R_max = std::max(R_max, l2(pts[static_cast<size_t>(i)], anchor_));
    } else
#endif
    {
      for (int i = 0; i < n; ++i) R_max = std::max(R_max, l2(pts[static_cast<size_t>(i)], anchor_));
    }
    auto t1 = clock::now();
    prof_.insertA_other_ms += msBetween(t0, t1);
    while (deltas_[L_] < 2.0 * R_max - 1e-15) expandTop();
    R_seen_ = R_max;

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

    coords_.reserve(coords_.size() + static_cast<uint32_t>(n));
    std::vector<uint32_t> cells(static_cast<size_t>(n), 0);
    t0 = clock::now();
    for (int i = 0; i < n; ++i) {
      cells[static_cast<size_t>(i)] = getOrCreateExactHashed(
          pts[static_cast<size_t>(i)].data(), ch1[static_cast<size_t>(i)], ch2[static_cast<size_t>(i)]);
    }
    t1 = clock::now();
    prof_.insertA_coord_io_ms += msBetween(t0, t1);

    const uint32_t recs_old = recs_live_;
    const uint32_t recs_cap = recs_old + static_cast<uint32_t>(n) * static_cast<uint32_t>(L_ + 1);
    if (recs_.size() < recs_cap) recs_.resize(recs_cap);
    std::atomic<uint32_t> recs_next{recs_old};

    const int nlay = L_ + 1;
    std::vector<uint32_t> path_pack(static_cast<size_t>(n) * static_cast<size_t>(nlay), 0);
    t0 = clock::now();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
      internLayerPathLocked(h_pack.data() + static_cast<size_t>(i) * static_cast<size_t>(K_),
                            hh1[static_cast<size_t>(i)], hh2[static_cast<size_t>(i)], recs_next,
                            path_pack.data() + static_cast<size_t>(i) * static_cast<size_t>(nlay));
    }
    t1 = clock::now();
    prof_.insertA_resolve_ms += msBetween(t0, t1);

    recs_live_ = recs_next.load();
    t0 = clock::now();
    for (uint32_t id = recs_old; id < recs_live_; ++id) {
      layer_ids_[static_cast<size_t>(recs_[id].layer)].push_back(id);
      const uint32_t p = recs_[id].parent;
      if (!p) continue;
      recs_[id].next_sibling = recs_[p].first_child;
      recs_[p].first_child = id;
    }
    t1 = clock::now();
    prof_.insertA_other_ms += msBetween(t0, t1);

    std::vector<PointId> ids;
    ids.reserve(static_cast<size_t>(n));
    t0 = clock::now();
    for (int i = 0; i < n; ++i) {
      ids.push_back(insertAMembers(
          cells[static_cast<size_t>(i)],
          path_pack.data() + static_cast<size_t>(i) * static_cast<size_t>(nlay)));
    }
    t1 = clock::now();
    prof_.insertA_posting_ms += msBetween(t0, t1);

    t0 = clock::now();
    rebuildStrata();
    t1 = clock::now();
    prof_.insertA_sampler_ms += msBetween(t0, t1);
    return ids;
  }

  PointId insertB(const std::vector<double>& x) {
    using clock = std::chrono::high_resolution_clock;
    ensureDim(x);
    ensureCapacityB();
    double e0 = prof_.expandTop_ms;
    auto t0 = clock::now();
    updateRangeAndExpand(x);
    auto t1 = clock::now();
    prof_.insertB_other_ms += msBetween(t0, t1) - (prof_.expandTop_ms - e0);

    PointId id = next_b_id_++;
    BRec rec;
    rec.id = id;
    if (b_by_id_.size() <= id) b_by_id_.resize(id + 1);
    ++b_live_;

    t0 = clock::now();
    fillH0(x.data(), h0_scratch_);
    t1 = clock::now();
    prof_.insertB_h0_ms += msBetween(t0, t1);

    t0 = clock::now();
    uint32_t cell = getOrCreateExact(x.data());
    rec.exact_id = cell;
    bool exact_b_zero = (exact_[cell].B_count == 0);
    exact_[cell].B_count += 1;
    if (exact_b_zero) syncExactStratum(cell);
    t1 = clock::now();
    prof_.insertB_exact_sampler_ms += msBetween(t0, t1);

    rec.membership.resize(static_cast<size_t>(L_) + 1);
    t0 = clock::now();
    resolveLayerPath(h0_scratch_);
    t1 = clock::now();
    prof_.insertB_resolve_ms += msBetween(t0, t1);
    t0 = clock::now();
    for (int i = 0; i <= L_; ++i) {
      uint32_t buck = resolve_path_[static_cast<size_t>(i)];
      rec.membership[static_cast<size_t>(i)] = buck;
      bool was_zero = (recs_[buck].B_count == 0);
      recs_[buck].B_count += 1;
      if (was_zero) onBucketBTransition(buck);
    }
    chillPathKeys(resolve_path_);
    t1 = clock::now();
    prof_.insertB_exact_sampler_ms += msBetween(t0, t1);

    t0 = clock::now();
    rec.nn_pos = static_cast<uint32_t>(nn_B_.size());
    nn_B_.push_back(id);
    {
      size_t off = B_flat_.size();
      B_flat_.resize(off + static_cast<size_t>(dim_));
      for (int d = 0; d < dim_; ++d) B_flat_[off + static_cast<size_t>(d)] = x[static_cast<size_t>(d)];
    }
    b_pool_.push_back(std::move(rec));
    b_by_id_[id] = static_cast<uint32_t>(b_pool_.size());  // 1-based index into b_pool_
    t1 = clock::now();
    prof_.insertB_other_ms += msBetween(t0, t1);
    ++prof_.insertB_count;
    return id;
  }

  void deleteB(PointId id) {
    using clock = std::chrono::high_resolution_clock;
    BRec& rec = bRec(id);
    auto t0 = clock::now();
    uint32_t cell = rec.exact_id;
    if (!cell) throw std::runtime_error("deleteB: missing exact cell");
    exact_[cell].B_count -= 1;
    if (exact_[cell].B_count == 0) syncExactStratum(cell);
    auto t1 = clock::now();
    prof_.deleteB_exact_sampler_ms += msBetween(t0, t1);

    t0 = clock::now();
    if (static_cast<int>(rec.membership.size()) != L_ + 1)
      throw std::runtime_error("deleteB: stale B membership");
    for (int i = 0; i <= L_; ++i) {
      uint32_t buck = rec.membership[static_cast<size_t>(i)];
      if (!buck) throw std::runtime_error("deleteB: missing bucket");
      recs_[buck].B_count -= 1;
      if (recs_[buck].B_count == 0) onBucketBTransition(buck);
    }
    t1 = clock::now();
    prof_.deleteB_membership_ms += msBetween(t0, t1);

    t0 = clock::now();
    size_t pos = rec.nn_pos;
    size_t last = nn_B_.size() - 1;
    if (pos != last) {
      PointId moved = nn_B_[last];
      nn_B_[pos] = moved;
      const size_t D = static_cast<size_t>(dim_);
      double* dst = B_flat_.data() + pos * D;
      const double* src = B_flat_.data() + last * D;
      for (size_t d = 0; d < D; ++d) dst[d] = src[d];
      bRec(moved).nn_pos = static_cast<uint32_t>(pos);
    }
    nn_B_.pop_back();
    B_flat_.resize(nn_B_.size() * static_cast<size_t>(dim_));
    b_by_id_[id] = 0;
    --b_live_;
    t1 = clock::now();
    prof_.deleteB_other_ms += msBetween(t0, t1);
    ++prof_.deleteB_count;
  }

  // 与 ChamferEstimator::queryStratified 同一估计量；点坐标从磁盘 gather。注释见内存版。
  double queryStratified(int T = -1) {
    using clock = std::chrono::high_resolution_clock;
    auto t_setup0 = clock::now();
    if (a_live_ == 0) return 0.0;
    if (nn_B_.empty()) return std::numeric_limits<double>::infinity();
    if (T <= 0) T = defaultT(a_live_);
    finalizeA();

    const int nS = L_ + 2;
    ensureStratumSize();
    std::vector<double> W(static_cast<size_t>(nS), 0.0);
    std::vector<int> N(static_cast<size_t>(nS), 0);
    for (int i = 0; i <= L_; ++i) {
      N[static_cast<size_t>(i)] = stratum_N_[static_cast<size_t>(i)];
      W[static_cast<size_t>(i)] = static_cast<double>(N[static_cast<size_t>(i)]) * deltas_[static_cast<size_t>(i)];
    }
    N[static_cast<size_t>(L_ + 1)] = stratum_N_[static_cast<size_t>(L_ + 1)];
    W[static_cast<size_t>(L_ + 1)] =
        static_cast<double>(N[static_cast<size_t>(L_ + 1)]) * 2.0 * deltas_[static_cast<size_t>(L_)];
    double Dalloc = 0.0;
    for (double w : W) Dalloc += w;
    auto t_setup1 = clock::now();
    prof_.query_overhead_ms += msBetween(t_setup0, t_setup1);
    if (!(Dalloc > 0.0)) {
      ++prof_.query_calls;
      return 0.0;
    }

    std::vector<int> Ti = hamiltonAllocateGuaranteed(W, N, T);
    struct SampleTask {
      PointId aid;
      int multiplicity;
    };
    std::vector<SampleTask> tasks;
    tasks.reserve(static_cast<size_t>(T));
    std::vector<std::vector<PointId>> draws(static_cast<size_t>(nS));
    std::vector<char> is_full(static_cast<size_t>(nS), 0);
    std::vector<int> task_lo(static_cast<size_t>(nS), 0);
    std::vector<int> task_hi(static_cast<size_t>(nS), 0);

    auto t_samp0 = clock::now();
    const auto io0 = ioSnap();
    for (int si = 0; si < nS; ++si) {
      const int ni = N[static_cast<size_t>(si)];
      int ti = Ti[static_cast<size_t>(si)];
      if (ni <= 0 || ti <= 0) continue;
      task_lo[static_cast<size_t>(si)] = static_cast<int>(tasks.size());
      if (ti >= ni) {
        is_full[static_cast<size_t>(si)] = 1;
        collectStratumAIds(si, tasks);
      } else {
        auto& dr = draws[static_cast<size_t>(si)];
        dr.resize(static_cast<size_t>(ti));
        std::vector<uint32_t> bkt, slot, exact_e, exact_s;
        std::vector<int> kind;  // 0 exact, 1 bucket
        bkt.reserve(static_cast<size_t>(ti));
        slot.reserve(static_cast<size_t>(ti));
        kind.reserve(static_cast<size_t>(ti));
        for (int t = 0; t < ti; ++t) {
          ComponentId cid = stratum_samp_[static_cast<size_t>(si)].sample(rng_);
          uint32_t idx = unpackId(cid.ptr);
          if (cid.kind == ComponentKind::Exact) {
            if (exact_[idx].A_count <= 0) throw std::runtime_error("empty exact members");
            uint32_t s = static_cast<uint32_t>(
                std::uniform_int_distribution<int>(0, exact_[idx].A_count - 1)(rng_));
            kind.push_back(0);
            exact_e.push_back(idx);
            exact_s.push_back(s);
            bkt.push_back(0);
            slot.push_back(0);
          } else {
            if (recs_[idx].A_count <= 0) throw std::runtime_error("empty bucket A_count");
            uint32_t s = static_cast<uint32_t>(
                std::uniform_int_distribution<int>(0, recs_[idx].A_count - 1)(rng_));
            kind.push_back(1);
            bkt.push_back(idx);
            slot.push_back(s);
            exact_e.push_back(0);
            exact_s.push_back(0);
          }
        }
        std::vector<PointId> aids(static_cast<size_t>(ti), 0);
        std::vector<uint32_t> gb, gs;
        std::vector<int> gpos;
        for (int t = 0; t < ti; ++t) {
          if (kind[static_cast<size_t>(t)] == 0) {
            aids[static_cast<size_t>(t)] = exactAt(exact_e[static_cast<size_t>(t)], exact_s[static_cast<size_t>(t)]);
          } else {
            gb.push_back(bkt[static_cast<size_t>(t)]);
            gs.push_back(slot[static_cast<size_t>(t)]);
            gpos.push_back(t);
          }
        }
        if (!gb.empty()) {
          std::vector<PointId> got(gb.size(), 0);
          postings_.gather(gb.data(), gs.data(), static_cast<int>(gb.size()), got.data());
          for (size_t j = 0; j < gpos.size(); ++j) aids[static_cast<size_t>(gpos[j])] = got[j];
        }
        std::unordered_map<PointId, int> mult;
        mult.reserve(static_cast<size_t>(ti));
        for (int t = 0; t < ti; ++t) {
          dr[static_cast<size_t>(t)] = aids[static_cast<size_t>(t)];
          ++mult[aids[static_cast<size_t>(t)]];
        }
        for (const auto& kv : mult) tasks.push_back(SampleTask{kv.first, kv.second});
      }
      task_hi[static_cast<size_t>(si)] = static_cast<int>(tasks.size());
    }
    auto t_samp1 = clock::now();
    prof_.query_sample_ms += msBetween(t_samp0, t_samp1);

    const int ntasks = static_cast<int>(tasks.size());
    std::vector<double> da_out(static_cast<size_t>(ntasks), 0.0);
    std::vector<double> coord_scratch(static_cast<size_t>(ntasks) * static_cast<size_t>(dim_));
    auto t_io0 = clock::now();
    if (ntasks > 0) {
      std::vector<uint32_t> cidx(static_cast<size_t>(ntasks));
      for (int i = 0; i < ntasks; ++i) {
        PointId aid = tasks[static_cast<size_t>(i)].aid;
        if (aid >= a_by_id_.size() || a_by_id_[aid].id == 0)
          throw std::runtime_error("queryStratified: unknown A id");
        cidx[static_cast<size_t>(i)] = a_by_id_[aid].coord_idx;
      }
      coords_.gather(cidx.data(), ntasks, coord_scratch.data());
    }
    auto t_io1 = clock::now();
    prof_.query_io_ms += msBetween(t_io0, t_io1);
    const auto io1 = ioSnap();
    prof_.io_reads += io1.n_read - io0.n_read;
    prof_.io_writes += io1.n_write - io0.n_write;
    prof_.io_bytes_read += static_cast<int64_t>(io1.bytes_read - io0.bytes_read);

    auto t_nn0 = clock::now();
#if defined(_OPENMP)
    if (ntasks >= 64) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < ntasks; ++i) {
        da_out[static_cast<size_t>(i)] =
            exactNN(coord_scratch.data() + static_cast<size_t>(i) * static_cast<size_t>(dim_));
      }
    } else
#endif
    {
      for (int i = 0; i < ntasks; ++i) {
        da_out[static_cast<size_t>(i)] =
            exactNN(coord_scratch.data() + static_cast<size_t>(i) * static_cast<size_t>(dim_));
      }
    }
    auto t_nn1 = clock::now();
    prof_.query_nn_ms += msBetween(t_nn0, t_nn1);

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
        Chat += sum_d;
        prof_.query_samples += ni;
      } else {
        std::unordered_map<PointId, double> da_map;
        da_map.reserve(static_cast<size_t>(hi - lo));
        for (int i = lo; i < hi; ++i) da_map[tasks[static_cast<size_t>(i)].aid] = da_out[static_cast<size_t>(i)];
        double sum_d = 0.0;
        for (PointId aid : draws[static_cast<size_t>(si)]) sum_d += da_map[aid];
        Chat += (static_cast<double>(ni) / static_cast<double>(ti)) * sum_d;
        prof_.query_samples += ti;
      }
    }
    ++prof_.query_calls;
    return Chat;
  }

  // 与 ChamferEstimator::queryImportance 同一估计量。层内抽点后批量读 posting / 坐标。
  double queryImportance(int T = -1) {
    using clock = std::chrono::high_resolution_clock;
    auto t_setup0 = clock::now();
    if (a_live_ == 0) return 0.0;
    if (nn_B_.empty()) return std::numeric_limits<double>::infinity();
    if (T <= 0) T = defaultT(a_live_);
    finalizeA();

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

    std::vector<int> draw_si(static_cast<size_t>(T), 0);
    std::vector<PointId> draw_aid(static_cast<size_t>(T), 0);
    auto t_samp0 = clock::now();
    const auto io0 = ioSnap();
    std::vector<uint32_t> gb, gs;
    std::vector<int> gpos;
    gb.reserve(static_cast<size_t>(T));
    gs.reserve(static_cast<size_t>(T));
    gpos.reserve(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t) {
      const int si = drawStratumProportional(W, D, rng_);
      draw_si[static_cast<size_t>(t)] = si;
      ComponentId cid = stratum_samp_[static_cast<size_t>(si)].sample(rng_);
      const uint32_t idx = unpackId(cid.ptr);
      if (cid.kind == ComponentKind::Exact) {
        if (exact_[idx].A_count <= 0) throw std::runtime_error("queryImportance: empty exact");
        const uint32_t s = static_cast<uint32_t>(
            std::uniform_int_distribution<int>(0, exact_[idx].A_count - 1)(rng_));
        draw_aid[static_cast<size_t>(t)] = exactAt(idx, s);
      } else {
        if (recs_[idx].A_count <= 0) throw std::runtime_error("queryImportance: empty bucket");
        const uint32_t s = static_cast<uint32_t>(
            std::uniform_int_distribution<int>(0, recs_[idx].A_count - 1)(rng_));
        gb.push_back(idx);
        gs.push_back(s);
        gpos.push_back(t);
      }
    }
    if (!gb.empty()) {
      std::vector<PointId> got(gb.size(), 0);
      postings_.gather(gb.data(), gs.data(), static_cast<int>(gb.size()), got.data());
      for (size_t j = 0; j < gpos.size(); ++j)
        draw_aid[static_cast<size_t>(gpos[j])] = got[j];
    }
    std::unordered_map<PointId, int> mult;
    mult.reserve(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t) {
      if (draw_aid[static_cast<size_t>(t)] == 0)
        throw std::runtime_error("queryImportance: unresolved A id");
      ++mult[draw_aid[static_cast<size_t>(t)]];
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

    const int ntasks = static_cast<int>(tasks.size());
    std::vector<double> da_out(static_cast<size_t>(ntasks), 0.0);
    std::vector<double> coord_scratch(static_cast<size_t>(ntasks) * static_cast<size_t>(dim_));
    auto t_io0 = clock::now();
    if (ntasks > 0) {
      std::vector<uint32_t> cidx(static_cast<size_t>(ntasks));
      for (int i = 0; i < ntasks; ++i) {
        const PointId aid = tasks[static_cast<size_t>(i)].aid;
        if (aid >= a_by_id_.size() || a_by_id_[aid].id == 0)
          throw std::runtime_error("queryImportance: unknown A id");
        cidx[static_cast<size_t>(i)] = a_by_id_[aid].coord_idx;
      }
      coords_.gather(cidx.data(), ntasks, coord_scratch.data());
    }
    auto t_io1 = clock::now();
    prof_.query_io_ms += msBetween(t_io0, t_io1);
    const auto io1 = ioSnap();
    prof_.io_reads += io1.n_read - io0.n_read;
    prof_.io_writes += io1.n_write - io0.n_write;
    prof_.io_bytes_read += static_cast<int64_t>(io1.bytes_read - io0.bytes_read);

    auto t_nn0 = clock::now();
#if defined(_OPENMP)
    if (ntasks >= 64) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < ntasks; ++i) {
        da_out[static_cast<size_t>(i)] =
            exactNN(coord_scratch.data() + static_cast<size_t>(i) * static_cast<size_t>(dim_));
      }
    } else
#endif
    {
      for (int i = 0; i < ntasks; ++i) {
        da_out[static_cast<size_t>(i)] =
            exactNN(coord_scratch.data() + static_cast<size_t>(i) * static_cast<size_t>(dim_));
      }
    }
    auto t_nn1 = clock::now();
    prof_.query_nn_ms += msBetween(t_nn0, t_nn1);

    std::unordered_map<PointId, double> da_map;
    da_map.reserve(static_cast<size_t>(ntasks));
    for (int i = 0; i < ntasks; ++i)
      da_map[tasks[static_cast<size_t>(i)].aid] = da_out[static_cast<size_t>(i)];
    double sumX = 0.0;
    for (int t = 0; t < T; ++t) {
      const int si = draw_si[static_cast<size_t>(t)];
      const double sc = scale[static_cast<size_t>(si)];
      sumX += D * da_map[draw_aid[static_cast<size_t>(t)]] / sc;
    }
    prof_.query_samples += T;
    ++prof_.query_calls;
    return sumX / static_cast<double>(T);
  }

  int firstCollisionLayer(PointId id) const {
    const ARec& arec = aRec(id);
    if (exact_[arec.exact_id].B_count > 0) return -2;
    uint32_t b = exact_[arec.exact_id].l0_bucket;
    for (int i = 0; i <= L_; ++i) {
      if (!b) return -1;
      if (recs_[b].B_count > 0) return i;
      b = recs_[b].parent;
    }
    return -1;
  }

  int stratumN(int si) const {
    ensureStratumSizeConst();
    if (si < 0 || si >= static_cast<int>(stratum_N_.size())) return 0;
    return stratum_N_[static_cast<size_t>(si)];
  }
  int stratumCount() const { return L_ + 2; }
  int fallbackStratumIndex() const { return L_ + 1; }
  int sumStratumN() const {
    ensureStratumSizeConst();
    int s = 0;
    for (int v : stratum_N_) s += v;
    return s;
  }
  int countPositiveDistanceA() const {
    int s = 0;
    for (PointId id = 1; id < a_by_id_.size(); ++id) {
      if (a_by_id_[id].id == 0) continue;
      if (exact_[a_by_id_[id].exact_id].B_count == 0) ++s;
    }
    return s;
  }
  bool validateStrataInvariant(double* sum_n_out = nullptr, double* pos_out = nullptr) const {
    int sn = sumStratumN();
    int pos = countPositiveDistanceA();
    if (sum_n_out) *sum_n_out = sn;
    if (pos_out) *pos_out = pos;
    return sn == pos;
  }
  std::mt19937_64 rngSnapshot() const { return rng_; }
  void rngRestore(const std::mt19937_64& s) { rng_ = s; }
  PointId sampleStratumPublic(int si) { return sampleAFromStratum(si); }

  std::vector<PointId> allAIds() const {
    std::vector<PointId> ids;
    ids.reserve(a_live_);
    for (PointId id = 1; id < a_by_id_.size(); ++id)
      if (a_by_id_[id].id != 0) ids.push_back(id);
    return ids;
  }

  double bruteSumDa() const {
    double s = 0.0;
    for (PointId id = 1; id < a_by_id_.size(); ++id) {
      if (a_by_id_[id].id == 0) continue;
      if (exact_[a_by_id_[id].exact_id].B_count > 0) continue;
      int j = firstCollisionLayer(id);
      if (j == -2) continue;
      if (j < 0) s += 2.0 * deltas_[static_cast<size_t>(L_)];
      else s += deltas_[static_cast<size_t>(j)];
    }
    return s;
  }

 private:
  struct BucketRec {
    uint32_t parent = 0;
    uint32_t first_child = 0;
    uint32_t next_sibling = 0;
    int32_t A_count = 0;
    int32_t B_count = 0;
    uint64_t key_hash = 0;
    uint64_t key_hash2 = 0;
    int16_t layer = 0;
    int16_t stratum_idx = -1;
    int32_t stratum_dense_pos = -1;
    SamplerSlotRef stratum_slot;
    uint8_t in_stratum = 0;
  };
  struct ExactRec {
    uint32_t coord_idx = 0;
    uint32_t l0_bucket = 0;
    int32_t A_count = 0;
    int32_t B_count = 0;
    uint64_t hash = 0;
    uint64_t hash2 = 0;
    int16_t stratum_idx = -1;
    int32_t stratum_dense_pos = -1;
    SamplerSlotRef stratum_slot;
    uint8_t in_stratum = 0;
    uint64_t sync_stamp = 0;
  };
  struct ARec {
    PointId id = 0;
    uint32_t coord_idx = 0;
    uint32_t exact_id = 0;
    uint32_t exact_pos = 0;
  };
  struct BRec {
    PointId id = 0;
    uint32_t exact_id = 0;
    uint32_t nn_pos = 0;
    std::vector<uint32_t> membership;
  };
  static double msBetween(std::chrono::high_resolution_clock::time_point a,
                          std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  }
  static void* packId(uint32_t id) { return reinterpret_cast<void*>(static_cast<uintptr_t>(id)); }
  static uint32_t unpackId(void* p) { return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p)); }

  storage::IoStats ioSnap() const {
    storage::IoStats s = coords_.file().stats();
    const auto& p = postings_.file().stats();
    s.n_read += p.n_read;
    s.n_write += p.n_write;
    s.bytes_read += p.bytes_read;
    s.bytes_write += p.bytes_write;
    s.read_ms += p.read_ms;
    s.write_ms += p.write_ms;
    return s;
  }
  void snapIoStats() {
    const auto s = ioSnap();
    prof_.io_reads = static_cast<int64_t>(s.n_read);
    prof_.io_writes = static_cast<int64_t>(s.n_write);
    prof_.io_bytes_read = static_cast<int64_t>(s.bytes_read);
  }

  void ensureDim(const std::vector<double>& x) const {
    if (static_cast<int>(x.size()) != dim_) throw std::invalid_argument("dimension mismatch");
  }
  void ensureCapacityA() {
    if (static_cast<int>(a_live_) + 1 > n_max_) throw std::runtime_error("|A| would exceed n_max");
  }
  void ensureCapacityB() {
    if (static_cast<int>(b_live_) + 1 > n_max_) throw std::runtime_error("|B| would exceed n_max");
  }

  const ARec& aRec(PointId id) const {
    if (id >= a_by_id_.size() || a_by_id_[id].id == 0) throw std::runtime_error("unknown A id");
    return a_by_id_[id];
  }
  BRec& bRec(PointId id) {
    if (id >= b_by_id_.size() || b_by_id_[id] == 0) throw std::runtime_error("unknown B id");
    return b_pool_[static_cast<size_t>(b_by_id_[id] - 1)];
  }

  void fillH0(const double* xd, BucketKey& key) const {
    e2lsh::fillH0(xd, dim_, K_, g_flat_.data(), S0_.data(), rho_, key);
  }
  void parentKeyInPlace(BucketKey& z, int i) const {
    e2lsh::parentKeyInPlace(z, K_, xi_[static_cast<size_t>(i)]);
  }
  BucketKey parentKey(const BucketKey& z, int i) const {
    BucketKey p = z;
    parentKeyInPlace(p, i);
    return p;
  }

  void updateRangeAndExpand(const std::vector<double>& x) {
    if (!has_anchor_) {
      anchor_ = x;
      has_anchor_ = true;
      R_seen_ = 0.0;
    } else {
      R_seen_ = std::max(R_seen_, l2(x, anchor_));
    }
    while (deltas_[L_] < 2.0 * R_seen_ - 1e-15) expandTop();
  }

  PointId insertAStructure(const std::vector<double>& x, BucketKey& h0) {
    using clock = std::chrono::high_resolution_clock;
    PointId id = next_a_id_++;
    if (a_by_id_.size() <= id) a_by_id_.resize(id + 1);
    ARec rec;
    rec.id = id;
    ++a_live_;

    auto t0 = clock::now();
    uint32_t cell = getOrCreateExact(x.data());
    rec.coord_idx = exact_[cell].coord_idx;
    rec.exact_id = cell;
    rec.exact_pos = ex_len_[cell];
    exactAppend(cell, id);
    exact_[cell].A_count += 1;
    auto t1 = clock::now();
    prof_.insertA_exact_ms += msBetween(t0, t1);

    t0 = clock::now();
    resolveLayerPath(h0);
    t1 = clock::now();
    prof_.insertA_resolve_ms += msBetween(t0, t1);
    for (int i = 0; i <= L_; ++i) {
      uint32_t buck = resolve_path_[static_cast<size_t>(i)];
      recs_[buck].A_count += 1;
      postings_.append(buck, id);
    }
    exact_[cell].l0_bucket = resolve_path_[0];
    syncExactStratum(cell);
    syncPathFrontierBucket(resolve_path_);
    chillPathKeys(resolve_path_);
    a_by_id_[id] = rec;
    ++prof_.insertA_count;
    return id;
  }

  uint32_t getOrCreateExactHashed(const double* xd, std::size_t h1, std::size_t h2) {
    uint32_t id = exact_map_.find(h1, h2, [&](uint32_t e) { return exact_[e].hash2; });
    if (id) return id;
    uint32_t cidx = coords_.append(xd);
    id = static_cast<uint32_t>(exact_.size());
    ExactRec rec;
    rec.coord_idx = cidx;
    rec.hash = h1;
    rec.hash2 = h2;
    exact_.push_back(rec);
    ex_off_.push_back(0);
    ex_len_.push_back(0);
    ex_cap_.push_back(0);
    exact_map_.insert(h1, id);
    return id;
  }

  uint32_t getOrCreateExact(const double* xd) {
    std::size_t h1 = 0, h2 = 0;
    hashCoords(xd, dim_, h1, h2);
    return getOrCreateExactHashed(xd, h1, h2);
  }

  PointId insertAMembers(uint32_t cell, const uint32_t* path) {
    PointId id = next_a_id_++;
    if (a_by_id_.size() <= id) a_by_id_.resize(id + 1);
    ARec rec;
    rec.id = id;
    rec.coord_idx = exact_[cell].coord_idx;
    rec.exact_id = cell;
    rec.exact_pos = ex_len_[cell];
    exactAppend(cell, id);
    exact_[cell].A_count += 1;
    for (int i = 0; i <= L_; ++i) {
      uint32_t buck = path[i];
      recs_[buck].A_count += 1;
      postings_.append(buck, id);
    }
    exact_[cell].l0_bucket = path[0];
    chillPathKeys(path);
    a_by_id_[id] = rec;
    ++a_live_;
    ++prof_.insertA_count;
    return id;
  }

  uint32_t internBucketLocked(int layer, uint64_t h1, uint64_t h2, const int64_t* h,
                              std::atomic<uint32_t>& recs_next) {
    return maps_[static_cast<size_t>(layer)].findOrCreate(
        h1, h2, [&](uint32_t bid) { return recs_[bid].key_hash2; },
        [this, layer, h1, h2, h, &recs_next]() {
          const uint32_t id = recs_next.fetch_add(1);
          if (id >= recs_.size()) throw std::runtime_error("internBucketLocked: recs_ overflow");
          BucketRec rec;
          rec.layer = static_cast<int16_t>(layer);
          rec.key_hash = h1;
          rec.key_hash2 = h2;
          recs_[id] = rec;
          if (keep_warm_keys_ && layer >= L_) {
            std::lock_guard<std::mutex> lock(recs_alloc_mu_);
            warm_keys_[id].assign(h, h + K_);
          }
          return id;
        });
  }

  // Follow parent pointers from `from_layer` to L. Incomplete chain (parent==0) → false.
  // Safe under parallel intern: a zero is a missed shortcut, never a wrong parent.
  bool tryFillParentChain(int from_layer, uint32_t node, uint32_t* path) const {
    if (from_layer >= L_) return true;
    uint32_t x = node;
    for (int j = from_layer + 1; j <= L_; ++j) {
      x = recs_[x].parent;
      if (!x) return false;
      path[j] = x;
    }
    return true;
  }

  void internLayerPathLocked(int64_t* h_row, uint64_t h1, uint64_t h2,
                             std::atomic<uint32_t>& recs_next, uint32_t* path) {
    uint32_t prev = 0;
    for (int i = 0; i <= L_; ++i) {
      const uint32_t buck = internBucketLocked(i, h1, h2, h_row, recs_next);
      if (prev) recs_[prev].parent = buck;
      path[i] = buck;
      prev = buck;
      if (i < L_) {
        if (tryFillParentChain(i, buck, path)) return;
        parentKeyRowInPlace(h_row, K_, xi_[static_cast<size_t>(i)], h1, h2);
      }
    }
  }

  void clearStratumFlags() {
    const int nS = static_cast<int>(stratum_active_.size());
    for (int si = 0; si < nS; ++si) {
      for (const ComponentId& cid : stratum_active_[static_cast<size_t>(si)]) {
        const uint32_t idx = unpackId(cid.ptr);
        if (cid.kind == ComponentKind::Exact) {
          exact_[idx].in_stratum = 0;
          exact_[idx].stratum_idx = -1;
          exact_[idx].stratum_dense_pos = -1;
          exact_[idx].stratum_slot = SamplerSlotRef{};
        } else {
          recs_[idx].in_stratum = 0;
          recs_[idx].stratum_idx = -1;
          recs_[idx].stratum_dense_pos = -1;
          recs_[idx].stratum_slot = SamplerSlotRef{};
        }
      }
    }
  }

  void rebuildChildListsFromParents() {
    for (uint32_t b = 1; b < recs_live_; ++b) {
      recs_[b].first_child = 0;
      recs_[b].next_sibling = 0;
    }
    for (uint32_t b = 1; b < recs_live_; ++b) {
      const uint32_t p = recs_[b].parent;
      if (!p) continue;
      recs_[b].next_sibling = recs_[p].first_child;
      recs_[p].first_child = b;
    }
  }

  void rebuildStrata() {
    ensureStratumSize();
    const int nS = L_ + 2;
    clearStratumFlags();
    stratum_samp_.assign(static_cast<size_t>(nS), DynamicWeightedSampler{});
    stratum_N_.assign(static_cast<size_t>(nS), 0);
    stratum_active_.assign(static_cast<size_t>(nS), {});
    if (b_live_ == 0) {
      for (uint32_t b : layer_ids_[static_cast<size_t>(L_)])
        setBucketStratum(b, desiredBucketStratum(b));
      return;
    }
    for (size_t i = 1; i < exact_.size(); ++i)
      setExactStratum(static_cast<uint32_t>(i), desiredExactStratum(static_cast<uint32_t>(i)));
    for (uint32_t b = 1; b < recs_live_; ++b) setBucketStratum(b, desiredBucketStratum(b));
  }

  void exactAppend(uint32_t e, PointId id) {
    if (ex_len_[e] == ex_cap_[e]) {
      const uint32_t ncap = ex_cap_[e] == 0 ? 1u : ex_cap_[e] * 2u;
      const uint32_t noff = static_cast<uint32_t>(ex_arena_.size());
      ex_arena_.resize(static_cast<size_t>(noff) + ncap, 0);
      if (ex_len_[e] > 0)
        std::memcpy(ex_arena_.data() + noff, ex_arena_.data() + ex_off_[e],
                    static_cast<size_t>(ex_len_[e]) * sizeof(PointId));
      ex_off_[e] = noff;
      ex_cap_[e] = ncap;
    }
    ex_arena_[ex_off_[e] + ex_len_[e]] = id;
    ex_len_[e] += 1;
  }
  PointId exactAt(uint32_t e, uint32_t i) const { return ex_arena_[ex_off_[e] + i]; }

  void resolveLayerPath(BucketKey& cur) {
    resolve_path_.assign(static_cast<size_t>(L_) + 1, 0);
    uint32_t prev = 0;
    bool via_parent = false;
    for (int i = 0; i <= L_; ++i) {
      uint32_t buck = 0;
      if (via_parent) {
        if (!prev || !recs_[prev].parent) throw std::runtime_error("resolveLayerPath: missing parent");
        buck = recs_[prev].parent;
      } else {
        buck = findBucket(i, cur);
        if (buck) via_parent = true;
        else buck = createBucket(i, cur);
      }
      if (prev) linkParent(prev, buck);
      resolve_path_[static_cast<size_t>(i)] = buck;
      prev = buck;
      if (!via_parent && i < L_) parentKeyInPlace(cur, i);
    }
  }

  uint32_t findBucket(int layer, const BucketKey& key) const {
    return maps_[static_cast<size_t>(layer)].find(key.hash, key.hash2,
                                                 [&](uint32_t id) { return recs_[id].key_hash2; });
  }

  uint32_t createBucket(int layer, const BucketKey& key) {
    const uint32_t id = recs_live_;
    if (id >= recs_.size()) recs_.emplace_back();
    ++recs_live_;
    BucketRec rec;
    rec.layer = static_cast<int16_t>(layer);
    rec.key_hash = key.hash;
    rec.key_hash2 = key.hash2;
    recs_[id] = rec;
    postings_.ensureBucket(id);
    maps_[static_cast<size_t>(layer)].insert(key.hash, id);
    layer_ids_[static_cast<size_t>(layer)].push_back(id);
    if (keep_warm_keys_ && layer >= L_) warm_keys_[id] = key.h;
    return id;
  }

  uint32_t getOrCreateBucket(int layer, const BucketKey& key) {
    uint32_t e = findBucket(layer, key);
    if (e) return e;
    return createBucket(layer, key);
  }

  void chillPathKeys(const uint32_t* path) {
    for (int i = 0; i <= L_; ++i) {
      const uint32_t b = path[i];
      if (!b) continue;
      if (recs_[b].layer >= L_) continue;
      warm_keys_.erase(b);
    }
  }
  void chillPathKeys(const std::vector<uint32_t>& path) {
    if (!path.empty()) chillPathKeys(path.data());
  }

  BucketKey keyOfWarm(uint32_t b) {
    auto it = warm_keys_.find(b);
    if (it != warm_keys_.end()) {
      BucketKey k;
      k.k = K_;
      k.h = it->second;
      k.hash = recs_[b].key_hash;
      k.hash2 = recs_[b].key_hash2;
      return k;
    }
    BucketKey k;
    if (recs_[b].layer == 0) {
      if (postings_.length(b) == 0)
        throw std::runtime_error("keyOfWarm: L0 empty posting");
      PointId aid = postings_.at(b, 0);
      const ARec& arec = aRec(aid);
      if (coord_scratch_.size() != static_cast<size_t>(dim_)) coord_scratch_.resize(static_cast<size_t>(dim_));
      coords_.readOne(arec.coord_idx, coord_scratch_.data());
      fillH0(coord_scratch_.data(), k);
    } else {
      uint32_t child = recs_[b].first_child;
      if (!child) throw std::runtime_error("keyOfWarm: no child to reconstruct");
      BucketKey ck = keyOfWarm(child);
      k = parentKey(ck, recs_[b].layer - 1);
    }
    if (k.hash != recs_[b].key_hash)
      throw std::runtime_error("keyOfWarm: reconstructed hash mismatch");
    if (keep_warm_keys_) warm_keys_[b] = k.h;
    return k;
  }

  void expandTop() {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    std::bernoulli_distribution bern(0.5);
    std::vector<uint8_t> bits(static_cast<size_t>(K_));
    for (int j = 0; j < K_; ++j) bits[static_cast<size_t>(j)] = bern(rng_) ? 1 : 0;
    xi_.push_back(bits);
    int oldL = L_;
    int newL = oldL + 1;
    deltas_.push_back(deltas_[static_cast<size_t>(oldL)] * 2.0);
    maps_.emplace_back();
    layer_ids_.emplace_back();

    for (uint32_t child : layer_ids_[static_cast<size_t>(oldL)]) {
      BucketKey ck = keyOfWarm(child);
      BucketKey pk = parentKey(ck, oldL);
      uint32_t parent = getOrCreateBucket(newL, pk);
      linkParent(child, parent);
      recs_[parent].A_count += recs_[child].A_count;
      recs_[parent].B_count += recs_[child].B_count;
      if (recs_[child].A_count > 0) {
        std::vector<PointId> ids;
        postings_.readAll(child, ids);
        for (PointId aid : ids) postings_.append(parent, aid);
      }
    }

    for (PointId id = 1; id < b_by_id_.size(); ++id) {
      if (b_by_id_[id] == 0) continue;
      BRec& brec = bRec(id);
      if (static_cast<int>(brec.membership.size()) != oldL + 1)
        throw std::runtime_error("expandTop: stale B membership");
      uint32_t child = brec.membership[static_cast<size_t>(oldL)];
      if (!recs_[child].parent) throw std::runtime_error("expandTop: missing parent");
      brec.membership.resize(static_cast<size_t>(newL) + 1);
      brec.membership[static_cast<size_t>(newL)] = recs_[child].parent;
    }

    L_ = newL;
    ensureStratumSize();
    for (uint32_t b : layer_ids_[static_cast<size_t>(oldL)]) syncBucketStratum(b);
    for (uint32_t b : layer_ids_[static_cast<size_t>(newL)]) syncBucketStratum(b);
    for (uint32_t b : layer_ids_[static_cast<size_t>(oldL)]) warm_keys_.erase(b);

    auto t1 = clock::now();
    prof_.expandTop_ms += msBetween(t0, t1);
    ++prof_.expandTop_count;
  }

  void linkParent(uint32_t child, uint32_t parent) {
    if (!child) return;
    if (recs_[child].parent == parent) return;
    unlinkFromParent(child);
    recs_[child].parent = parent;
    if (parent) {
      recs_[child].next_sibling = recs_[parent].first_child;
      recs_[parent].first_child = child;
    }
  }

  void unlinkFromParent(uint32_t child) {
    if (!child || !recs_[child].parent) {
      if (child) {
        recs_[child].parent = 0;
        recs_[child].next_sibling = 0;
      }
      return;
    }
    uint32_t parent = recs_[child].parent;
    uint32_t prev = 0;
    uint32_t c = recs_[parent].first_child;
    while (c && c != child) {
      prev = c;
      c = recs_[c].next_sibling;
    }
    if (c == child) {
      if (prev) recs_[prev].next_sibling = recs_[child].next_sibling;
      else recs_[parent].first_child = recs_[child].next_sibling;
    }
    recs_[child].parent = 0;
    recs_[child].next_sibling = 0;
  }

  int desiredBucketStratum(uint32_t b) const {
    if (!b || recs_[b].A_count <= 0 || recs_[b].B_count != 0) return -1;
    if (recs_[b].layer == L_) return L_ + 1;
    uint32_t p = recs_[b].parent;
    if (p && recs_[p].B_count > 0) return recs_[b].layer + 1;
    return -1;
  }
  int desiredExactStratum(uint32_t cell) const {
    if (!cell || exact_[cell].A_count <= 0 || exact_[cell].B_count != 0) return -1;
    uint32_t l0 = exact_[cell].l0_bucket;
    if (l0 && recs_[l0].B_count > 0) return 0;
    return -1;
  }

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

  void stratumActiveRemove(int idx, int pos, ComponentKind kind) {
    if (idx < 0 || idx >= static_cast<int>(stratum_active_.size())) return;
    auto& v = stratum_active_[static_cast<size_t>(idx)];
    if (pos < 0 || pos >= static_cast<int>(v.size())) return;
    int last = static_cast<int>(v.size()) - 1;
    if (pos != last) {
      ComponentId moved = v[static_cast<size_t>(last)];
      v[static_cast<size_t>(pos)] = moved;
      uint32_t mid = unpackId(moved.ptr);
      if (moved.kind == ComponentKind::Exact) exact_[mid].stratum_dense_pos = pos;
      else recs_[mid].stratum_dense_pos = pos;
    }
    v.pop_back();
    (void)kind;
  }

  void clearBucketStratum(uint32_t b) {
    if (!b || !recs_[b].in_stratum) return;
    ensureStratumSize();
    int idx = recs_[b].stratum_idx;
    if (idx >= 0 && idx < static_cast<int>(stratum_samp_.size())) {
      int w = static_cast<int>(
          std::lround(stratum_samp_[static_cast<size_t>(idx)].weightAt(recs_[b].stratum_slot)));
      stratum_N_[static_cast<size_t>(idx)] -= w;
      if (stratum_N_[static_cast<size_t>(idx)] < 0) stratum_N_[static_cast<size_t>(idx)] = 0;
      stratum_samp_[static_cast<size_t>(idx)].destroy(recs_[b].stratum_slot);
      stratumActiveRemove(idx, recs_[b].stratum_dense_pos, ComponentKind::Bucket);
    }
    recs_[b].in_stratum = 0;
    recs_[b].stratum_idx = -1;
    recs_[b].stratum_dense_pos = -1;
    recs_[b].stratum_slot = SamplerSlotRef{};
  }

  void clearExactStratum(uint32_t cell) {
    if (!cell || !exact_[cell].in_stratum) return;
    ensureStratumSize();
    int idx = exact_[cell].stratum_idx;
    if (idx >= 0 && idx < static_cast<int>(stratum_samp_.size())) {
      int w = static_cast<int>(
          std::lround(stratum_samp_[static_cast<size_t>(idx)].weightAt(exact_[cell].stratum_slot)));
      stratum_N_[static_cast<size_t>(idx)] -= w;
      if (stratum_N_[static_cast<size_t>(idx)] < 0) stratum_N_[static_cast<size_t>(idx)] = 0;
      stratum_samp_[static_cast<size_t>(idx)].destroy(exact_[cell].stratum_slot);
      stratumActiveRemove(idx, exact_[cell].stratum_dense_pos, ComponentKind::Exact);
    }
    exact_[cell].in_stratum = 0;
    exact_[cell].stratum_idx = -1;
    exact_[cell].stratum_dense_pos = -1;
    exact_[cell].stratum_slot = SamplerSlotRef{};
  }

  void setBucketStratum(uint32_t b, int idx) {
    ensureStratumSize();
    if (!b || idx < 0) {
      clearBucketStratum(b);
      return;
    }
    const double w = static_cast<double>(recs_[b].A_count);
    if (recs_[b].in_stratum && recs_[b].stratum_idx == idx) {
      double oldw = stratum_samp_[static_cast<size_t>(idx)].weightAt(recs_[b].stratum_slot);
      stratum_samp_[static_cast<size_t>(idx)].setWeight(recs_[b].stratum_slot, w);
      stratum_N_[static_cast<size_t>(idx)] += static_cast<int>(std::lround(w - oldw));
      return;
    }
    clearBucketStratum(b);
    ComponentId cid;
    cid.kind = ComponentKind::Bucket;
    cid.layer = recs_[b].layer;
    cid.ptr = packId(b);
    recs_[b].stratum_slot = stratum_samp_[static_cast<size_t>(idx)].create(w, cid);
    recs_[b].in_stratum = 1;
    recs_[b].stratum_idx = static_cast<int16_t>(idx);
    recs_[b].stratum_dense_pos = static_cast<int32_t>(stratum_active_[static_cast<size_t>(idx)].size());
    stratum_active_[static_cast<size_t>(idx)].push_back(cid);
    stratum_N_[static_cast<size_t>(idx)] += recs_[b].A_count;
  }

  void setExactStratum(uint32_t cell, int idx) {
    ensureStratumSize();
    if (!cell || idx < 0) {
      clearExactStratum(cell);
      return;
    }
    const double w = static_cast<double>(exact_[cell].A_count);
    if (exact_[cell].in_stratum && exact_[cell].stratum_idx == idx) {
      double oldw = stratum_samp_[static_cast<size_t>(idx)].weightAt(exact_[cell].stratum_slot);
      stratum_samp_[static_cast<size_t>(idx)].setWeight(exact_[cell].stratum_slot, w);
      stratum_N_[static_cast<size_t>(idx)] += static_cast<int>(std::lround(w - oldw));
      return;
    }
    clearExactStratum(cell);
    ComponentId cid;
    cid.kind = ComponentKind::Exact;
    cid.ptr = packId(cell);
    exact_[cell].stratum_slot = stratum_samp_[static_cast<size_t>(idx)].create(w, cid);
    exact_[cell].in_stratum = 1;
    exact_[cell].stratum_idx = static_cast<int16_t>(idx);
    exact_[cell].stratum_dense_pos = static_cast<int32_t>(stratum_active_[static_cast<size_t>(idx)].size());
    stratum_active_[static_cast<size_t>(idx)].push_back(cid);
    stratum_N_[static_cast<size_t>(idx)] += exact_[cell].A_count;
  }

  void syncBucketStratum(uint32_t b) {
    if (!b) return;
    setBucketStratum(b, desiredBucketStratum(b));
  }
  void syncExactStratum(uint32_t cell) {
    if (!cell) return;
    setExactStratum(cell, desiredExactStratum(cell));
  }

  void syncPathFrontierBucket(const std::vector<uint32_t>& path) {
    if (path.empty()) return;
    int j = L_ + 1;
    for (int i = 0; i <= L_; ++i) {
      if (recs_[path[static_cast<size_t>(i)]].B_count > 0) {
        j = i;
        break;
      }
    }
    if (j == 0) return;
    if (j <= L_) syncBucketStratum(path[static_cast<size_t>(j - 1)]);
    else syncBucketStratum(path[static_cast<size_t>(L_)]);
  }

  void syncExactCellsTouchingL0(uint32_t l0) {
    if (!l0 || recs_[l0].layer != 0) return;
    const uint64_t g = ++sync_gen_;
    if (sync_gen_ == 0) {
      sync_gen_ = 1;
      for (size_t i = 1; i < exact_.size(); ++i) exact_[i].sync_stamp = 0;
    }
    std::vector<PointId> ids;
    postings_.readAll(l0, ids);
    for (PointId aid : ids) {
      if (aid >= a_by_id_.size() || a_by_id_[aid].id == 0) continue;
      uint32_t cell = a_by_id_[aid].exact_id;
      if (exact_[cell].sync_stamp == g) continue;
      exact_[cell].sync_stamp = g;
      syncExactStratum(cell);
    }
  }

  void onBucketBTransition(uint32_t b) {
    if (!b) return;
    syncBucketStratum(b);
    uint32_t c = recs_[b].first_child;
    while (c) {
      if (recs_[c].A_count > 0) syncBucketStratum(c);
      c = recs_[c].next_sibling;
    }
    if (recs_[b].layer == 0) syncExactCellsTouchingL0(b);
  }

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

  PointId sampleAFromStratum(int si) {
    ensureStratumSize();
    if (si < 0 || si >= static_cast<int>(stratum_samp_.size()))
      throw std::runtime_error("sampleAFromStratum: bad stratum");
    if (!(stratum_samp_[static_cast<size_t>(si)].totalWeight() > 0.0))
      throw std::runtime_error("sampleAFromStratum: empty stratum");
    ComponentId cid = stratum_samp_[static_cast<size_t>(si)].sample(rng_);
    uint32_t idx = unpackId(cid.ptr);
    if (cid.kind == ComponentKind::Exact) {
      if (exact_[idx].A_count <= 0) throw std::runtime_error("empty exact members");
      uint32_t s =
          static_cast<uint32_t>(std::uniform_int_distribution<int>(0, exact_[idx].A_count - 1)(rng_));
      return exactAt(idx, s);
    }
    if (recs_[idx].A_count <= 0) throw std::runtime_error("empty bucket A_count");
    uint32_t s =
        static_cast<uint32_t>(std::uniform_int_distribution<int>(0, recs_[idx].A_count - 1)(rng_));
    return postings_.at(idx, s);
  }

  template <typename TaskVec>
  void collectStratumAIds(int si, TaskVec& tasks) {
    ensureStratumSize();
    if (si < 0 || si >= static_cast<int>(stratum_active_.size())) return;
    for (const ComponentId& cid : stratum_active_[static_cast<size_t>(si)]) {
      uint32_t idx = unpackId(cid.ptr);
      if (cid.kind == ComponentKind::Exact) {
        for (uint32_t i = 0; i < ex_len_[idx]; ++i)
          tasks.push_back(typename TaskVec::value_type{exactAt(idx, i), 1});
      } else {
        std::vector<PointId> ids;
        postings_.readAll(idx, ids);
        for (PointId aid : ids) tasks.push_back(typename TaskVec::value_type{aid, 1});
      }
    }
  }

  double exactNN(const double* x) const {
    return simd_l2::min_l2_flat(x, dim_, B_flat_.data(), nn_B_.size());
  }

  ProfileBreakdown prof_;
  int dim_;
  double rho_;
  int n_max_;
  int K_;
  std::mt19937_64 rng_;
  std::string storage_dir_;

  std::vector<double> g_flat_;
  std::vector<double> S0_;
  std::vector<std::vector<uint8_t>> xi_;
  int L_ = 0;
  std::vector<double> deltas_;
  bool has_anchor_ = false;
  std::vector<double> anchor_;
  double R_seen_ = 0.0;

  mutable std::vector<DynamicWeightedSampler> stratum_samp_;
  mutable std::vector<int> stratum_N_;
  mutable std::vector<std::vector<ComponentId>> stratum_active_;
  mutable uint64_t sync_gen_ = 1;

  std::vector<BucketRec> recs_;
  uint32_t recs_live_ = 1;  // valid ids are 1 .. recs_live_-1; recs_.size() may be larger (spare)
  std::vector<ShardedFingerMap> maps_;
  std::vector<std::vector<uint32_t>> layer_ids_;
  std::unordered_map<uint32_t, std::vector<int64_t>> warm_keys_;
  bool keep_warm_keys_ = true;
  std::vector<double> coord_scratch_;
  std::vector<uint32_t> resolve_path_;
  BucketKey h0_scratch_;

  std::vector<ExactRec> exact_;
  ShardedFingerMap exact_map_;
  std::mutex recs_alloc_mu_;
  std::vector<PointId> ex_arena_;
  std::vector<uint32_t> ex_off_, ex_len_, ex_cap_;

  std::vector<ARec> a_by_id_;
  std::size_t a_live_ = 0;
  std::vector<BRec> b_pool_;
  std::vector<uint32_t> b_by_id_;  // 1-based index into b_pool_, 0 = none
  std::size_t b_live_ = 0;
  std::vector<PointId> nn_B_;
  std::vector<double, AlignedAlloc<double>> B_flat_;
  PointId next_a_id_ = 1;
  PointId next_b_id_ = 1;

  storage::CoordStore coords_;
  storage::PostingStore postings_;
};

}  // namespace e2lsh
