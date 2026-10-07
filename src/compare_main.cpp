#include "e2lsh_chamfer.hpp"
#include "e2lsh_chamfer_external.hpp"
#include "csv_io.hpp"
#include "os/mapped_file.hpp"

#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>
#if defined(_WIN32)
#include <malloc.h>
#endif

// Vendor baseline (copy under vendor/)
#include "quad_tree.h"
#include "aspect_ratio_depth.h"
#include "exact_nn.hpp"
#include "datatype.h"
#include "fast_dynamic_chamfer.hpp"
#include "dataset_config.h"
#include "random.h"

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

// AVX-512 loads are 64 bytes. A 16-byte-aligned window makes every load split a
// cache line, and on d=128 that scan is about 2x slower than E2LSH's buffer.
struct AlignedDoubles {
  double* p = nullptr;
  void reset(std::size_t n) {
#if defined(_WIN32)
    _aligned_free(p);
    p = n ? static_cast<double*>(_aligned_malloc(n * sizeof(double), 64)) : nullptr;
#else
    std::free(p);
    const std::size_t bytes = (n * sizeof(double) + 63u) & ~std::size_t{63};
    p = n ? static_cast<double*>(std::aligned_alloc(64, bytes)) : nullptr;
#endif
    if (n && !p) throw std::bad_alloc();
  }
  ~AlignedDoubles() {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
  AlignedDoubles() = default;
  AlignedDoubles(const AlignedDoubles&) = delete;
  AlignedDoubles& operator=(const AlignedDoubles&) = delete;
  double* data() const { return p; }
};

static thread_local std::mt19937 gen(42);

// QuadTree / Uniform exact NN. Same kernel as ChamferEstimator::exactNN.
static double sample_uniform_ids_flat(const std::vector<std::vector<double>>& A,
                                      const std::deque<size_t>& ids, int sample_times,
                                      const double* Bflat, int dim, size_t nB) {
  if (ids.empty() || sample_times <= 0 || nB == 0) return 0.0;
  std::uniform_int_distribution<size_t> dist(0, ids.size() - 1);
  double sum = 0.0;
  for (int i = 0; i < sample_times; ++i) {
    const auto& q = A[ids[dist(gen)]];
    sum += e2lsh::simd_l2::min_l2_flat(q.data(), dim, Bflat, nB);
  }
  return sum * static_cast<double>(ids.size()) / static_cast<double>(sample_times);
}

// Unidirectional Uniform: sample the full static A, then the shared flat scan.
static double sample_uniform_flat(const std::vector<std::vector<double>>& A,
                                   int sample_times, const double* Bflat, int dim, size_t nB) {
  if (A.empty() || sample_times <= 0 || nB == 0 || Bflat == nullptr) return 0.0;
  std::uniform_int_distribution<size_t> dist(0, A.size() - 1);
  double sum = 0.0;
  for (int i = 0; i < sample_times; ++i) {
    const auto& q = A[dist(gen)];
    sum += e2lsh::simd_l2::min_l2_flat(q.data(), dim, Bflat, nB);
  }
  return sum * static_cast<double>(A.size()) / static_cast<double>(sample_times);
}

static double sample_quadtree_flat(QuadTree& tree, const std::vector<std::vector<double>>& A,
                                   int sample_times, const double* Bflat, int dim, size_t nB) {
  (void)A;
  double sum = 0.0;
  int got = 0;
  int guard = 0;
  std::vector<double> q(static_cast<size_t>(dim));
  while (got < sample_times && guard < sample_times * 1000) {
    ++guard;
    int32_t node = tree.sample_by_weight();
    if (node < 0) continue;
    const double width = tree.node_width(node);
    while (node >= 0) {
      const int p = tree.node_single_A(node);
      if (p >= 0) {
        tree.copy_A_row(p, q.data());
        double dist = e2lsh::simd_l2::min_l2_flat(q.data(), dim, Bflat, nB);
        sum += dist * tree.total_weight / width;
        ++got;
        break;
      }
      node = tree.sample_node_by_weight(node);
    }
  }
  if (got == 0) return 0.0;
  return sum / sample_times;
}

struct NnMicro {
  int reps = 0;
  double e2_ms = 0.0;
  double flat_ms = 0.0;
  double ratio = 1.0;
  std::size_t e2_nB = 0;
  std::size_t flat_nB = 0;
  const char* status = "SKIP";
  const char* note = "";
};

// Time E2LSH exactNN against the shared flat kernel on the same queries.
// Grow the repeat count until each side is past the clock grain, then apply a 2x gate.
static NnMicro nn_microbench(e2lsh::ChamferEstimator& est,
                             const std::vector<std::vector<double>>& A,
                             const std::deque<size_t>& live_A,
                             const double* Bflat, int dim, size_t nB) {
  NnMicro m;
  m.e2_nB = est.sizeB();
  m.flat_nB = nB;
  if (live_A.empty() || nB == 0 || Bflat == nullptr) {
    std::cout << "NN_MICROBENCH status=SKIP\n";
    return m;
  }
  if (est.sizeB() != nB) {
    m.status = "FAIL";
    m.note = "nB_mismatch";
    std::cout << "NN_MICROBENCH status=FAIL reason=nB_mismatch e2_nB=" << est.sizeB()
              << " flat_nB=" << nB << "\n";
    return m;
  }
  auto q_at = [&](int r) -> const std::vector<double>& {
    return A[live_A[static_cast<size_t>(r) % live_A.size()]];
  };
  double sink = 0.0;
  for (int r = 0; r < 4; ++r) {
    sink += est.exactNNPublic(q_at(r));
    sink += e2lsh::simd_l2::min_l2_flat(q_at(r).data(), dim, Bflat, nB);
  }
  int reps = 32;
  double e2_ms = 0.0;
  double flat_ms = 0.0;
  for (int attempt = 0; attempt < 8; ++attempt) {
    auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
      sink += e2lsh::simd_l2::min_l2_flat(q_at(r).data(), dim, Bflat, nB);
    flat_ms = Duration(Clock::now() - t0).count();
    t0 = Clock::now();
    for (int r = 0; r < reps; ++r) sink += est.exactNNPublic(q_at(r));
    e2_ms = Duration(Clock::now() - t0).count();
    if (std::min(e2_ms, flat_ms) >= 8.0 || reps >= 4096) break;
    reps *= 2;
  }
  const double lo = std::min(e2_ms, flat_ms);
  const double hi = std::max(e2_ms, flat_ms);
  const double ratio = (lo > 0.0) ? (hi / lo) : 1.0;
  const bool resolved = lo >= 5.0;
  m.reps = reps;
  m.e2_ms = e2_ms;
  m.flat_ms = flat_ms;
  m.ratio = ratio;
  m.note = resolved ? "timed" : "short_interval";
  m.status = (!resolved || ratio <= 2.0) ? "OK" : "FAIL";
  std::cout << std::setprecision(6);
  const auto e2_align =
      reinterpret_cast<std::uintptr_t>(est.nnBData()) & static_cast<std::uintptr_t>(63);
  const auto flat_align =
      reinterpret_cast<std::uintptr_t>(Bflat) & static_cast<std::uintptr_t>(63);
  std::cout << "NN_MICROBENCH reps=" << reps
            << " e2_ms=" << e2_ms << " flat_ms=" << flat_ms
            << " ratio_slow_over_fast=" << ratio
            << " e2_nB=" << m.e2_nB << " flat_nB=" << nB
            << " e2_align=" << e2_align << " flat_align=" << flat_align
            << " note=" << m.note << " status=" << m.status
            << " sink=" << sink << "\n";
  return m;
}

struct Args {
  std::string data_dir = "data/Fashion-MNIST";
  std::string out_csv = "output/compare_cost_log.csv";
  std::string timing_json = "output/timing_summary.json";
  std::string exact_cache_path;  // --exact-cache <path>; empty = always compute
  bool exact_cache_only = false;  // generate/verify cache without E2LSH
  bool smoke = false;
  bool e2lsh_only = false;
  bool quadtree_only = false;  // skip E2LSH (memory-split full SIFT)
  bool inject_outlier = false;  // append ã = mean(A) + 0.1|A|(A[0] - mean(A))
  bool no_outlier = false;      // load train.f32bin / base.f32bin; never inject
  bool dual_dynamic = false;    // A and B both slide
  bool with_is = false;         // --with-is: also run queryImportance (requires --e2lsh-only)
  int smoke_A = 800;
  int smoke_B = 400;
  uint64_t seed = 42;
  int T_override = -1;  // --T / --samples; <=0 means use computeDefaultT
  int n_max = -1;  // --n-max; <=0 → max(70000, |A|+window+1024)
  int max_A = -1;  // --max-A; truncate A after load/outlier (pilot)
  int max_B = -1;  // --max-B; truncate B
  int window_size = -1;  // --window-size; <=0 → |B|/20
  int query_interval = -1;  // --query-interval; <=0 → max(|B|/56, 1)
  std::string storage_dir;  // --storage-dir; empty = in-RAM ChamferEstimator
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--smoke") a.smoke = true;
    else if (s == "--e2lsh-only") a.e2lsh_only = true;
    else if (s == "--quadtree-only") a.quadtree_only = true;
    else if (s == "--data-dir" && i + 1 < argc) a.data_dir = argv[++i];
    else if (s == "--out" && i + 1 < argc) a.out_csv = argv[++i];
    else if (s == "--timing-out" && i + 1 < argc) a.timing_json = argv[++i];
    else if (s == "--smoke-A" && i + 1 < argc) a.smoke_A = std::stoi(argv[++i]);
    else if (s == "--smoke-B" && i + 1 < argc) a.smoke_B = std::stoi(argv[++i]);
    else if (s == "--seed" && i + 1 < argc) a.seed = std::stoull(argv[++i]);
    else if ((s == "--T" || s == "--samples") && i + 1 < argc) a.T_override = std::stoi(argv[++i]);
    else if (s == "--exact-cache" && i + 1 < argc) a.exact_cache_path = argv[++i];
    else if (s == "--exact-cache-only") a.exact_cache_only = true;
    else if (s == "--inject-outlier") a.inject_outlier = true;
    else if (s == "--no-outlier") a.no_outlier = true;
    else if (s == "--dual-dynamic") a.dual_dynamic = true;
    else if (s == "--with-is") a.with_is = true;
    else if (s == "--n-max" && i + 1 < argc) a.n_max = std::stoi(argv[++i]);
    else if (s == "--max-A" && i + 1 < argc) a.max_A = std::stoi(argv[++i]);
    else if (s == "--max-B" && i + 1 < argc) a.max_B = std::stoi(argv[++i]);
    else if ((s == "--window-size" || s == "--window") && i + 1 < argc)
      a.window_size = std::stoi(argv[++i]);
    else if ((s == "--query-interval" || s == "--query-every") && i + 1 < argc)
      a.query_interval = std::stoi(argv[++i]);
    else if (s == "--storage-dir" && i + 1 < argc)
      a.storage_dir = argv[++i];
  }
  return a;
}

// In-RAM or out-of-core estimator with a shared call surface for the driver.
struct E2 {
  std::unique_ptr<e2lsh::ChamferEstimator> ram;
  std::unique_ptr<e2lsh::ExternalChamferEstimator> ext;
  bool is_ext() const { return static_cast<bool>(ext); }
  int K() const { return ext ? ext->K() : ram->K(); }
  int L() const { return ext ? ext->L() : ram->L(); }
  double totalD() const { return ext ? ext->totalD() : ram->totalD(); }
  e2lsh::PointId insertA(const std::vector<double>& x) {
    return ext ? ext->insertA(x) : ram->insertA(x);
  }
  std::vector<e2lsh::PointId> insertABatch(const std::vector<std::vector<double>>& pts) {
    return ext ? ext->insertABatch(pts) : ram->insertABatch(pts);
  }
  e2lsh::PointId insertB(const std::vector<double>& x) {
    return ext ? ext->insertB(x) : ram->insertB(x);
  }
  void deleteB(e2lsh::PointId id) {
    if (ext) ext->deleteB(id);
    else ram->deleteB(id);
  }
  void deleteA(e2lsh::PointId id) {
    if (ext) throw std::runtime_error("deleteA is not supported on the external backend");
    ram->deleteA(id);
  }
  double queryStratified(int T) { return ext ? ext->queryStratified(T) : ram->queryStratified(T); }
  double queryImportance(int T) { return ext ? ext->queryImportance(T) : ram->queryImportance(T); }
  std::mt19937_64 rngSnapshot() const { return ext ? ext->rngSnapshot() : ram->rngSnapshot(); }
  void rngRestore(const std::mt19937_64& s) {
    if (ext) ext->rngRestore(s);
    else ram->rngRestore(s);
  }
  bool validateStrataInvariant() {
    return ext ? ext->validateStrataInvariant() : ram->validateStrataInvariant();
  }
  int sumStratumN() const { return ext ? ext->sumStratumN() : ram->sumStratumN(); }
  int countPositiveDistanceA() const {
    return ext ? ext->countPositiveDistanceA() : ram->countPositiveDistanceA();
  }
  const e2lsh::ProfileBreakdown& profile() const { return ext ? ext->profile() : ram->profile(); }
  void expandToCover(const std::vector<double>& x) {
    if (ext) ext->expandToCover(x);
  }
  void expandToCoverR(double R) {
    if (ext) ext->expandToCoverR(R);
  }
  void finalizeA() {
    if (ext) ext->finalizeA();
  }
};


static std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\': o += "\\\\"; break;
      case '"': o += "\\\""; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default: o += c; break;
    }
  }
  return o;
}

static bool file_exists_path(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return static_cast<bool>(in);
}

// Compact binary: uint64 n, uint64 d, then n*d float32 row-major (little-endian).
static std::vector<std::vector<double>> read_f32bin(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "Error opening f32bin: " << path << std::endl;
    return {};
  }
  uint64_t n = 0, d = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(n));
  in.read(reinterpret_cast<char*>(&d), sizeof(d));
  if (!in || n == 0 || d == 0 || d > 4096 || n > 5'000'000ULL) {
    std::cerr << "Bad f32bin header in " << path << " n=" << n << " d=" << d << std::endl;
    return {};
  }
  std::vector<float> buf(static_cast<size_t>(n * d));
  in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(float)));
  if (!in) {
    std::cerr << "Short read on f32bin " << path << std::endl;
    return {};
  }
  std::vector<std::vector<double>> data(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(d)));
  for (uint64_t i = 0; i < n; ++i) {
    for (uint64_t j = 0; j < d; ++j) {
      data[static_cast<size_t>(i)][static_cast<size_t>(j)] =
          static_cast<double>(buf[static_cast<size_t>(i * d + j)]);
    }
  }
  return data;
}

// Flat f32bin via file mapping (file-backed; not anon RSS) for E2LSH-only streaming insert.
struct FlatF32 {
  uint64_t n = 0, d = 0;
  const float* data = nullptr;  // n*d row-major
  e2lsh::os::MappedFile map;
  bool empty() const { return n == 0 || data == nullptr; }
  const float* row(size_t i) const { return data + i * static_cast<size_t>(d); }
  void reset() {
    map.close();
    n = 0;
    d = 0;
    data = nullptr;
  }
};

static FlatF32 read_f32bin_flat(const std::string& path) {
  FlatF32 out;
  if (!out.map.openRead(path)) {
    std::cerr << "Error opening f32bin: " << path << std::endl;
    return out;
  }
  if (out.map.size() < 16) {
    std::cerr << "Bad f32bin stat " << path << std::endl;
    out.reset();
    return out;
  }
  const auto* base = reinterpret_cast<const uint64_t*>(out.map.data());
  out.n = base[0];
  out.d = base[1];
  if (out.n == 0 || out.d == 0 || out.d > 4096 || out.n > 5'000'000ULL) {
    std::cerr << "Bad f32bin header in " << path << " n=" << out.n << " d=" << out.d << std::endl;
    out.reset();
    return out;
  }
  const size_t need = 16 + static_cast<size_t>(out.n * out.d) * sizeof(float);
  if (out.map.size() < need) {
    std::cerr << "Short f32bin " << path << std::endl;
    out.reset();
    return out;
  }
  out.data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(out.map.data()) + 16);
  return out;
}

static void promote_row(const float* src, int dim, std::vector<double>& dst) {
  dst.resize(static_cast<size_t>(dim));
  for (int j = 0; j < dim; ++j) dst[static_cast<size_t>(j)] = static_cast<double>(src[j]);
}

static double l2_f32(const float* u, const float* p, int dim) {
  double sum = 0.0;
  for (int j = 0; j < dim; ++j) {
    const double d = static_cast<double>(p[j]) - static_cast<double>(u[j]);
    sum += d * d;
  }
  return std::sqrt(sum);
}

static double l2_f32_f64(const float* u, const std::vector<double>& p, int dim) {
  double sum = 0.0;
  for (int j = 0; j < dim; ++j) {
    const double d = p[static_cast<size_t>(j)] - static_cast<double>(u[j]);
    sum += d * d;
  }
  return std::sqrt(sum);
}

// compute_2approx_diameter(A then B) is 2 * max L2 from point 0. No second copy of A.
static double diameter_2approx_nested(const std::vector<std::vector<double>>& A,
                                     const std::vector<std::vector<double>>& B) {
  if (A.empty()) return compute_2approx_diameter(B);
  if (B.empty()) return compute_2approx_diameter(A);
  const std::vector<double>& u = A[0];
  double max_dist = 0.0;
  for (const auto& p : A) max_dist = std::max(max_dist, euclidean_distance(u, p));
  for (const auto& p : B) max_dist = std::max(max_dist, euclidean_distance(u, p));
  return max_dist * 2.0;
}

static double diameter_2approx_flat(const FlatF32& A, const std::vector<std::vector<double>>& B) {
  if (A.empty()) return compute_2approx_diameter(B);
  const int dim = static_cast<int>(A.d);
  const float* u = A.row(0);
  double max_dist = 0.0;
  for (uint64_t i = 0; i < A.n; ++i)
    max_dist = std::max(max_dist, l2_f32(u, A.row(static_cast<size_t>(i)), dim));
  for (const auto& p : B) max_dist = std::max(max_dist, l2_f32_f64(u, p, dim));
  return max_dist * 2.0;
}

static double sample_uniform_ids_f32(const FlatF32& A, const std::deque<size_t>& ids, int sample_times,
                                     const double* Bflat, int dim, size_t nB) {
  if (ids.empty() || sample_times <= 0 || nB == 0) return 0.0;
  std::uniform_int_distribution<size_t> dist(0, ids.size() - 1);
  std::vector<double> q(static_cast<size_t>(dim));
  double sum = 0.0;
  for (int i = 0; i < sample_times; ++i) {
    promote_row(A.row(ids[dist(gen)]), dim, q);
    sum += e2lsh::simd_l2::min_l2_flat(q.data(), dim, Bflat, nB);
  }
  return sum * static_cast<double>(ids.size()) / static_cast<double>(sample_times);
}

static double sample_uniform_f32(const FlatF32& A, int sample_times, const double* Bflat, int dim, size_t nB) {
  if (A.empty() || sample_times <= 0 || nB == 0 || Bflat == nullptr) return 0.0;
  std::uniform_int_distribution<size_t> dist(0, static_cast<size_t>(A.n) - 1);
  std::vector<double> q(static_cast<size_t>(dim));
  double sum = 0.0;
  for (int i = 0; i < sample_times; ++i) {
    promote_row(A.row(dist(gen)), dim, q);
    sum += e2lsh::simd_l2::min_l2_flat(q.data(), dim, Bflat, nB);
  }
  return sum * static_cast<double>(A.n) / static_cast<double>(sample_times);
}

// Texmex .fvecs: per vector int32 dim + dim float32.
static std::vector<std::vector<double>> read_fvecs(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "Error opening fvecs: " << path << std::endl;
    return {};
  }
  std::vector<std::vector<double>> data;
  while (true) {
    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!in) break;
    if (d <= 0 || d > 4096) {
      std::cerr << "Bad fvecs dim " << d << " in " << path << std::endl;
      return {};
    }
    std::vector<float> row(static_cast<size_t>(d));
    in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(d * sizeof(float)));
    if (!in) {
      std::cerr << "Short fvecs read in " << path << std::endl;
      return {};
    }
    std::vector<double> rd(static_cast<size_t>(d));
    for (int j = 0; j < d; ++j) rd[static_cast<size_t>(j)] = static_cast<double>(row[static_cast<size_t>(j)]);
    data.push_back(std::move(rd));
  }
  return data;
}

// Outlier: ã = mean(A) + 0.1|A|(A[0] - mean(A)); append.
static void inject_sift_outlier(std::vector<std::vector<double>>& A) {
  if (A.empty()) return;
  const size_t n = A.size();
  const size_t d = A[0].size();
  std::vector<double> c(d, 0.0);
  for (const auto& row : A) {
    for (size_t j = 0; j < d; ++j) c[j] += row[j];
  }
  for (size_t j = 0; j < d; ++j) c[j] /= static_cast<double>(n);
  std::vector<double> a_tilde(d);
  const double scale = 0.1 * static_cast<double>(n);
  for (size_t j = 0; j < d; ++j) {
    a_tilde[j] = scale * (A[0][j] - c[j]) + c[j];
  }
  A.push_back(std::move(a_tilde));
  std::cout << "inject_outlier: |A| now " << A.size() << std::endl;
}

static std::vector<std::vector<double>> load_matrix_prefer(
    const std::string& data_dir,
    const std::vector<std::string>& names) {
  for (const auto& name : names) {
    const std::string path = data_dir + "/" + name;
    if (!file_exists_path(path)) continue;
    std::cout << "loading " << path << std::endl;
    if (name.size() >= 7 && name.substr(name.size() - 7) == ".f32bin") return read_f32bin(path);
    if (name.size() >= 6 && name.substr(name.size() - 6) == ".fvecs") return read_fvecs(path);
    if (name.size() >= 4 && name.substr(name.size() - 4) == ".csv") return read_csv_file(path);
  }
  return {};
}

static bool file_readable(const std::string& path) {
  std::ifstream in(path);
  return static_cast<bool>(in);
}

static std::map<size_t, double> load_exact_cache(const std::string& path) {
  std::map<size_t, double> m;
  std::ifstream in(path);
  if (!in) {
    std::cerr << "FATAL: cannot read exact cache " << path << std::endl;
    std::exit(1);
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (line.rfind("step", 0) == 0) continue;  // header
    const auto comma = line.find(',');
    if (comma == std::string::npos) continue;
    const size_t step = static_cast<size_t>(std::stoull(line.substr(0, comma)));
    const double chamfer = std::stod(line.substr(comma + 1));
    m[step] = chamfer;
  }
  return m;
}

static void write_exact_cache(const std::string& path,
                              const std::string& data_dir,
                              size_t nA, size_t nB,
                              int window_size, int query_interval,
                              const std::vector<std::pair<size_t, double>>& entries) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "FATAL: cannot write exact cache " << path << std::endl;
    std::exit(1);
  }
  out << "# exact Chamfer cache\n";
  out << "# data_dir=" << data_dir << "\n";
  out << "# |A|=" << nA << "\n";
  out << "# |B|=" << nB << "\n";
  out << "# window_size=" << window_size << "\n";
  out << "# query_interval=" << query_interval << "\n";
  out << "step,chamfer\n";
  out << std::setprecision(17);
  for (const auto& e : entries) {
    out << e.first << "," << e.second << "\n";
  }
}

static int run_dual_dynamic(const Args& args,
                            std::vector<std::vector<double>>& A,
                            std::vector<std::vector<double>>& B,
                            int dim, int window_size, int query_interval,
                            Clock::time_point wall_start,
                            const FlatF32* flatA) {
  if (!args.storage_dir.empty()) {
    std::cerr << "ERROR: --dual-dynamic is in-RAM only (no --storage-dir)\n";
    return 1;
  }
  // Wall-clock comparison: E2LSH queries and Benchmark updates are OpenMP,
  // QuadTree and Uniform are not. One thread makes the four timers comparable.
  omp_set_num_threads(1);
  const bool flat = flatA && !flatA->empty();
  const size_t nAs = flat ? static_cast<size_t>(flatA->n) : A.size();
  // The outlier is the last row of A. The sliding window would insert it
  // after the last query, so every estimate would be computed without it.
  // Pin it for the whole run: it stays in A and is not part of the window.
  const bool pin_outlier = !args.no_outlier;
  if (pin_outlier && nAs == 0) {
    std::cerr << "ERROR: with-outlier dual-dynamic needs an A point to pin\n";
    return 1;
  }
  const size_t nA_slide = pin_outlier ? nAs - 1 : nAs;
  const size_t outlier_idx = pin_outlier ? nAs - 1 : static_cast<size_t>(0);
  std::vector<double> a_row(static_cast<size_t>(dim));
  auto a_at = [&](size_t idx) -> const std::vector<double>& {
    if (!flat) return A[idx];
    promote_row(flatA->row(idx), dim, a_row);
    return a_row;
  };
  std::cout << "omp_threads=" << omp_get_max_threads() << "\n";
  std::cout << "dual-dynamic schedule=proportional |A|=" << nA_slide << " |B|=" << B.size()
            << " window=" << window_size << " query_interval=" << query_interval
            << (flat ? " (flat-A)" : "")
            << (pin_outlier ? " pinned_outlier=1" : "") << std::endl;

  const int n_max = args.quadtree_only
                        ? 8
                        : ((args.n_max > 0) ? args.n_max
                                            : std::max(70000, window_size + 1024));
  const double rho = 1e-4;

  std::unique_ptr<QuadTree> tree;
  double qt_init_insert_A_ms = 0.0;
  if (!args.e2lsh_only) {
    const double max_dist = flat ? diameter_2approx_flat(*flatA, B) : diameter_2approx_nested(A, B);
    double aspect_ratio = calculate_aspect_ratio(A, B, max_dist);
    int depth = calculate_quadtree_depth(aspect_ratio);
    std::cout << "diameter=" << max_dist << " aspect=" << aspect_ratio << " depth=" << depth
              << std::endl;
    auto t0 = Clock::now();
    tree = std::make_unique<QuadTree>(DIM, max_dist * 2, depth);
    if (flat) {
      CoordAccess acc;
      acc.Af = flatA->data;
      acc.nA = static_cast<int>(flatA->n);
      acc.Bn = &B;
      acc.nB = static_cast<int>(B.size());
      acc.dim = dim;
      tree->bind(acc);
      tree->build_from_bound();
    } else {
      std::vector<std::vector<double>> emptyA, emptyB;
      tree->build(A, B, emptyA, emptyB);
    }
    tree->Build_Tree_Sampler();
    qt_init_insert_A_ms = Duration(Clock::now() - t0).count();
  }

  E2 e2;
  e2.ram = std::make_unique<e2lsh::ChamferEstimator>(dim, rho, n_max, args.seed);
  if (!args.quadtree_only) {
    std::cout << "E2LSH K=" << e2.K() << " rho=" << rho << " n_max=" << n_max << std::endl;
  }

  FastDynamicChamfer dc(static_cast<std::size_t>(dim));

  struct WItem {
    char label;
    size_t idx;
    e2lsh::PointId pid;
  };
  std::deque<WItem> window;
  std::deque<size_t> live_A;
  std::deque<std::vector<double>> live_B;

  std::ofstream cost_log(args.out_csv);
  if (!cost_log.is_open()) {
    std::cerr << "Cannot open " << args.out_csv << std::endl;
    return 1;
  }
  if (args.quadtree_only)
    cost_log << "step,chamfer,quadtree_est,uniform_est,quadtree_err,uniform_err\n";
  else if (args.e2lsh_only)
    cost_log << "step,chamfer,strat_est,strat_err\n";
  else
    cost_log << "step,chamfer,quadtree_est,e2lsh_est,uniform_est,quadtree_err,e2lsh_err,uniform_err\n";

  double qt_update_ms_total = 0.0, e2_update_ms_total = 0.0;
  double qt_query_ms_total = 0.0, strat_query_ms_total = 0.0, uni_query_ms_total = 0.0;
  double exact_query_ms_total = 0.0, exact_update_ms_total = 0.0;
  int update_ops = 0, queries = 0;

  if (pin_outlier) {
    std::cout << "pinned_outlier idx=" << outlier_idx << " slide_nA=" << nA_slide << std::endl;
    if (!args.e2lsh_only) {
      auto t0 = Clock::now();
      tree->insert_id('A', static_cast<int>(outlier_idx));
      qt_update_ms_total += Duration(Clock::now() - t0).count();
    }
    if (!args.quadtree_only) {
      auto t0 = Clock::now();
      (void)e2.insertA(a_at(outlier_idx));
      e2_update_ms_total += Duration(Clock::now() - t0).count();
    }
    auto t0 = Clock::now();
    (void)dc.insert_A(a_at(outlier_idx));
    exact_update_ms_total += Duration(Clock::now() - t0).count();
  }

  size_t ai = 0, bi = 0;
  int a_run = 0, max_a_run = 0;
  std::string schedule_prefix;
  bool nn_checked = false;
  NnMicro nn_micro;
  size_t step = 0;
  const unsigned long long nA64 = static_cast<unsigned long long>(nA_slide);
  const unsigned long long nB64 = static_cast<unsigned long long>(B.size());
  while (ai < nA_slide || bi < B.size()) {
    char label;
    size_t idx;
    // Integer |A|:|B| reproduces "k insertions from A, then one from B".
    const bool take_A =
        ai < nA_slide &&
        (bi >= B.size() ||
         (static_cast<unsigned long long>(ai) + 1ULL) * nB64 <=
             (static_cast<unsigned long long>(bi) + 1ULL) * nA64);
    if (take_A) {
      label = 'A';
      idx = ai++;
      ++a_run;
      if (a_run > max_a_run) max_a_run = a_run;
    } else {
      label = 'B';
      idx = bi++;
      a_run = 0;
    }
    if (schedule_prefix.size() < 24) schedule_prefix.push_back(label);

    WItem item{label, idx, 0};
    if (label == 'A') {
      if (!args.e2lsh_only) {
        auto t0 = Clock::now();
        tree->insert_id('A', static_cast<int>(idx));
        qt_update_ms_total += Duration(Clock::now() - t0).count();
      }
      if (!args.quadtree_only) {
        auto t0 = Clock::now();
        item.pid = e2.insertA(a_at(idx));
        e2_update_ms_total += Duration(Clock::now() - t0).count();
      }
      auto t0 = Clock::now();
      (void)dc.insert_A(a_at(idx));
      exact_update_ms_total += Duration(Clock::now() - t0).count();
      live_A.push_back(idx);
    } else {
      if (!args.e2lsh_only) {
        auto t0 = Clock::now();
        tree->insert(B[idx], 'B', static_cast<int>(idx) + B_ID_shift, A, B);
        qt_update_ms_total += Duration(Clock::now() - t0).count();
      }
      if (!args.quadtree_only) {
        auto t0 = Clock::now();
        item.pid = e2.insertB(B[idx]);
        e2_update_ms_total += Duration(Clock::now() - t0).count();
      }
      auto t0 = Clock::now();
      dc.insert_B(B[idx]);
      exact_update_ms_total += Duration(Clock::now() - t0).count();
      live_B.push_back(B[idx]);
    }
    window.push_back(item);

    if (static_cast<int>(window.size()) > window_size) {
      const WItem old = window.front();
      window.pop_front();
      if (old.label == 'A') {
        if (!args.e2lsh_only) {
          auto t0 = Clock::now();
          tree->remove_id('A', static_cast<int>(old.idx));
          qt_update_ms_total += Duration(Clock::now() - t0).count();
        }
        if (!args.quadtree_only) {
          auto t0 = Clock::now();
          e2.deleteA(old.pid);
          e2_update_ms_total += Duration(Clock::now() - t0).count();
        }
        auto t0 = Clock::now();
        // Outlier occupies exact-A slot 0 for the whole run. Sliding deletes
        // remove the oldest unpinned A, which stays at index 1.
        dc.delete_A(pin_outlier ? 1 : 0);
        exact_update_ms_total += Duration(Clock::now() - t0).count();
        live_A.pop_front();
      } else {
        if (!args.e2lsh_only) {
          auto t0 = Clock::now();
          tree->remove(B[old.idx], 'B', A, B);
          qt_update_ms_total += Duration(Clock::now() - t0).count();
        }
        if (!args.quadtree_only) {
          auto t0 = Clock::now();
          e2.deleteB(old.pid);
          e2_update_ms_total += Duration(Clock::now() - t0).count();
        }
        auto t0 = Clock::now();
        dc.delete_B(0);
        exact_update_ms_total += Duration(Clock::now() - t0).count();
        live_B.pop_front();
      }
    }
    ++update_ops;

    if (static_cast<int>(step) >= window_size && step % static_cast<size_t>(query_interval) == 0 &&
        !live_A.empty() && !live_B.empty()) {
      const int n = static_cast<int>(live_A.size());
      const int T = (args.T_override > 0)
                        ? args.T_override
                        : e2lsh::ChamferEstimator::computeDefaultT(static_cast<size_t>(n));
      if (!args.quadtree_only && queries == 0) {
        if (!e2.validateStrataInvariant()) {
          std::cerr << "FATAL: strata invariant at first dual-dynamic query\n";
          return 2;
        }
      }
      if (pin_outlier && dc.size_A() != live_A.size() + 1) {
        std::cerr << "FATAL: pinned outlier missing at step " << step
                  << " exact_nA=" << dc.size_A() << " slide_nA=" << live_A.size() << "\n";
        return 2;
      }
      const size_t a_win = live_A.size() + (pin_outlier ? 1 : 0);
      std::cout << "Step " << step << " |Awin|=" << a_win << " |Bwin|=" << live_B.size()
                << " T=" << T << std::endl;

      // Harness copy of the live window. Not charged to any method.
      AlignedDoubles Bflat;
      if (!args.e2lsh_only) {
        Bflat.reset(live_B.size() * static_cast<size_t>(dim));
        std::size_t off = 0;
        for (const auto& b : live_B) {
          std::memcpy(Bflat.data() + off, b.data(), sizeof(double) * static_cast<size_t>(dim));
          off += static_cast<size_t>(dim);
        }
      }
      if (!nn_checked && !args.e2lsh_only && !args.quadtree_only) {
        nn_checked = true;
        if (flat) {
          std::vector<std::vector<double>> probeA;
          std::deque<size_t> probe;
          const size_t nprobe = std::min(live_A.size(), static_cast<size_t>(64));
          probeA.reserve(nprobe);
          for (size_t k = 0; k < nprobe; ++k) {
            probeA.push_back(a_at(live_A[k]));
            probe.push_back(k);
          }
          nn_micro = nn_microbench(*e2.ram, probeA, probe, Bflat.data(), dim, live_B.size());
        } else {
          nn_micro = nn_microbench(*e2.ram, A, live_A, Bflat.data(), dim, live_B.size());
        }
        if (std::strcmp(nn_micro.status, "FAIL") == 0) {
          std::cerr << "FATAL: exact-NN kernels disagree; refusing an unfair timing run\n";
          return 3;
        }
      }

      double chamfer = 0.0, qt_est = 0.0, uni_est = 0.0, strat_est = 0.0;
      {
        auto t0 = Clock::now();
        chamfer = dc.current();
        exact_query_ms_total += Duration(Clock::now() - t0).count();
      }
      if (pin_outlier && queries == 0) {
        const auto dists = dc.liveMinDists();
        std::cout << "pinned_outlier_check idx=" << outlier_idx
                  << " nn_dist=" << dists.front() << " chamfer=" << chamfer
                  << " exact_nA=" << dc.size_A() << std::endl;
      }
      if (!args.quadtree_only) {
        auto t0 = Clock::now();
        strat_est = e2.queryStratified(T);
        strat_query_ms_total += Duration(Clock::now() - t0).count();
      }
      if (!args.e2lsh_only) {
        {
          auto t0 = Clock::now();
          qt_est = sample_quadtree_flat(*tree, A, T, Bflat.data(), dim, live_B.size());
          qt_query_ms_total += Duration(Clock::now() - t0).count();
        }
        {
          auto t0 = Clock::now();
          if (pin_outlier) live_A.push_front(outlier_idx);
          uni_est = flat ? sample_uniform_ids_f32(*flatA, live_A, T, Bflat.data(), dim, live_B.size())
                        : sample_uniform_ids_flat(A, live_A, T, Bflat.data(), dim, live_B.size());
          if (pin_outlier) live_A.pop_front();
          uni_query_ms_total += Duration(Clock::now() - t0).count();
        }
      }

      if (args.quadtree_only) {
        const double qt_err = (qt_est - chamfer) / chamfer;
        const double uni_err = (uni_est - chamfer) / chamfer;
        std::cout << "  chamfer=" << chamfer << " qt=" << qt_est << " uni=" << uni_est << "\n";
        cost_log << step << "," << chamfer << "," << qt_est << "," << uni_est << "," << qt_err
                 << "," << uni_err << "\n";
      } else if (args.e2lsh_only) {
        const double strat_err = (strat_est - chamfer) / chamfer;
        std::cout << "  chamfer=" << chamfer << " strat=" << strat_est << "\n";
        cost_log << step << "," << chamfer << "," << strat_est << "," << strat_err << "\n";
      } else {
        const double qt_err = (qt_est - chamfer) / chamfer;
        const double e2_err = (strat_est - chamfer) / chamfer;
        const double uni_err = (uni_est - chamfer) / chamfer;
        std::cout << "  chamfer=" << chamfer << " qt=" << qt_est << " e2=" << strat_est
                  << " uni=" << uni_est << "\n";
        cost_log << step << "," << chamfer << "," << qt_est << "," << strat_est << "," << uni_est
                 << "," << qt_err << "," << e2_err << "," << uni_err << "\n";
      }
      cost_log.flush();
      ++queries;
    }
    ++step;
  }
  cost_log.close();

  const double wall_ms_total = Duration(Clock::now() - wall_start).count();
  const double qt_update_ms_avg = update_ops > 0 ? qt_update_ms_total / update_ops : 0.0;
  const double e2_update_ms_avg = update_ops > 0 ? e2_update_ms_total / update_ops : 0.0;
  const double qt_query_ms_avg = queries > 0 ? qt_query_ms_total / queries : 0.0;
  const double strat_query_ms_avg = queries > 0 ? strat_query_ms_total / queries : 0.0;
  const double uni_query_ms_avg = queries > 0 ? uni_query_ms_total / queries : 0.0;
  const double exact_query_ms_avg = queries > 0 ? exact_query_ms_total / queries : 0.0;
  const double exact_update_ms_avg = update_ops > 0 ? exact_update_ms_total / update_ops : 0.0;
  const double inv_ops = update_ops > 0 ? 1.0 / static_cast<double>(update_ops) : 0.0;
  // Same formula for every bar: structure update + amortized query.
  // QuadTree also amortizes the one-time empty-tree build. Diameter stays outside.
  const double qt_per_update_ms =
      qt_update_ms_avg + qt_query_ms_total * inv_ops + qt_init_insert_A_ms * inv_ops;
  const double e2_per_update_ms = e2_update_ms_avg + strat_query_ms_total * inv_ops;
  const double uni_per_update_ms = uni_query_ms_total * inv_ops;
  const double bench_per_update_ms = exact_update_ms_avg + exact_query_ms_total * inv_ops;

  std::cout << "schedule_done nA=" << nAs << " ai=" << ai << " nB=" << B.size()
            << " bi=" << bi << " max_A_run=" << max_a_run << " prefix=" << schedule_prefix
            << "\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\n========== TIMING SUMMARY (ms) dual-dynamic ==========\n";
  std::cout << "wall_ms_total=" << wall_ms_total << " queries=" << queries
            << " update_ops=" << update_ops << "\n";
  std::cout << "per_update_ms e2lsh=" << e2_per_update_ms << " quadtree=" << qt_per_update_ms
            << " uniform=" << uni_per_update_ms << " benchmark=" << bench_per_update_ms << "\n";
  if (!args.e2lsh_only) {
    std::cout << "QuadTree update_ms_avg=" << qt_update_ms_avg << " query_ms_avg=" << qt_query_ms_avg
              << "\n";
  }
  if (!args.quadtree_only) {
    std::cout << "E2LSH update_ms_avg=" << e2_update_ms_avg
              << " query_ms_avg=" << strat_query_ms_avg << "\n";
  }
  std::cout << "exact update_ms_avg=" << exact_update_ms_avg << "\n";
  std::cout << "=====================================================\n";

  std::ofstream tj(args.timing_json);
  if (!tj.is_open()) {
    std::cerr << "Cannot open " << args.timing_json << std::endl;
    return 1;
  }
  tj << std::fixed << std::setprecision(6);
  tj << "{\n";
  tj << "  \"seed\": " << args.seed << ",\n";
  tj << "  \"K\": " << e2.K() << ",\n";
  tj << "  \"L_final\": " << e2.L() << ",\n";
  tj << "  \"rho\": " << rho << ",\n";
  tj << "  \"n_max\": " << n_max << ",\n";
  tj << "  \"e2lsh_only\": " << (args.e2lsh_only ? "true" : "false") << ",\n";
  tj << "  \"quadtree_only\": " << (args.quadtree_only ? "true" : "false") << ",\n";
  tj << "  \"dual_dynamic\": true,\n";
  tj << "  \"no_outlier\": " << (args.no_outlier ? "true" : "false") << ",\n";
  tj << "  \"pinned_outlier\": " << (pin_outlier ? "true" : "false") << ",\n";
  tj << "  \"queries\": " << queries << ",\n";
  tj << "  \"T\": " << (args.T_override > 0 ? args.T_override : -1) << ",\n";
  tj << "  \"window_size\": " << window_size << ",\n";
  tj << "  \"query_interval\": " << query_interval << ",\n";
  tj << "  \"insert_schedule\": \"proportional\",\n";
  tj << "  \"omp_threads\": " << omp_get_max_threads() << ",\n";
  tj << "  \"nn_kernel\": \"simd_l2::min_l2_flat\",\n";
  tj << "  \"max_A_run\": " << max_a_run << ",\n";
  tj << "  \"schedule_prefix\": \"" << schedule_prefix << "\",\n";
  tj << "  \"nA\": " << nAs << ",\n";
  tj << "  \"nB\": " << B.size() << ",\n";
  tj << "  \"dim\": " << dim << ",\n";
  tj << "  \"update_ops\": " << update_ops << ",\n";
  tj << "  \"wall_ms_total\": " << wall_ms_total << ",\n";
  tj << "  \"quadtree\": {\n";
  tj << "    \"init_insert_A_ms\": " << qt_init_insert_A_ms << ",\n";
  tj << "    \"update_ms_total\": " << qt_update_ms_total << ",\n";
  tj << "    \"update_ms_avg\": " << qt_update_ms_avg << ",\n";
  tj << "    \"query_ms_total\": " << qt_query_ms_total << ",\n";
  tj << "    \"query_ms_avg\": " << qt_query_ms_avg << "\n";
  tj << "  },\n";
  tj << "  \"e2lsh\": {\n";
  tj << "    \"init_insert_A_ms\": 0.0,\n";
  tj << "    \"update_ms_total\": " << e2_update_ms_total << ",\n";
  tj << "    \"update_ms_avg\": " << e2_update_ms_avg << ",\n";
  tj << "    \"query_ms_total\": " << strat_query_ms_total << ",\n";
  tj << "    \"query_ms_avg\": " << strat_query_ms_avg << "\n";
  tj << "  },\n";
  tj << "  \"e2lsh_stratified\": {\n";
  tj << "    \"query_ms_total\": " << strat_query_ms_total << ",\n";
  tj << "    \"query_ms_avg\": " << strat_query_ms_avg << "\n";
  tj << "  },\n";
  tj << "  \"uniform\": {\n";
  tj << "    \"query_ms_total\": " << uni_query_ms_total << ",\n";
  tj << "    \"query_ms_avg\": " << uni_query_ms_avg << "\n";
  tj << "  },\n";
  tj << "  \"exact_chamfer\": {\n";
  tj << "    \"mode\": \"computed\",\n";
  tj << "    \"update_ms_total\": " << exact_update_ms_total << ",\n";
  tj << "    \"update_ms_avg\": " << exact_update_ms_avg << ",\n";
  tj << "    \"query_ms_total\": " << exact_query_ms_total << ",\n";
  tj << "    \"query_ms_avg\": " << exact_query_ms_avg << "\n";
  tj << "  },\n";
  tj << "  \"per_update_ms\": {\n";
  tj << "    \"E2LSH\": " << e2_per_update_ms << ",\n";
  tj << "    \"QuadTree\": " << qt_per_update_ms << ",\n";
  tj << "    \"Uniform\": " << uni_per_update_ms << ",\n";
  tj << "    \"Benchmark\": " << bench_per_update_ms << "\n";
  tj << "  },\n";
  tj << "  \"nn_microbench\": {\n";
  tj << "    \"reps\": " << nn_micro.reps << ",\n";
  tj << "    \"e2_ms\": " << nn_micro.e2_ms << ",\n";
  tj << "    \"flat_ms\": " << nn_micro.flat_ms << ",\n";
  tj << "    \"ratio_slow_over_fast\": " << nn_micro.ratio << ",\n";
  tj << "    \"e2_nB\": " << nn_micro.e2_nB << ",\n";
  tj << "    \"flat_nB\": " << nn_micro.flat_nB << ",\n";
  tj << "    \"note\": \"" << nn_micro.note << "\",\n";
  tj << "    \"status\": \"" << nn_micro.status << "\"\n";
  tj << "  }\n";
  tj << "}\n";
  std::cout << "Wrote timing summary to " << args.timing_json << std::endl;
  std::cout << "Done. queries=" << queries << " wrote " << args.out_csv << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  if (!args.storage_dir.empty()) {
    if (args.quadtree_only) {
      std::cerr << "ERROR: --storage-dir is mutually exclusive with --quadtree-only\n";
      return 1;
    }
    args.e2lsh_only = true;
  }
  if (args.e2lsh_only && args.quadtree_only) {
    std::cerr << "ERROR: --e2lsh-only and --quadtree-only are mutually exclusive\n";
    return 1;
  }
  if (args.with_is && !args.e2lsh_only) {
    std::cerr << "ERROR: --with-is requires --e2lsh-only\n";
    return 1;
  }
  if (args.with_is && args.dual_dynamic) {
    std::cerr << "ERROR: --with-is is not implemented on the dual-dynamic schedule\n";
    return 1;
  }
  // Seed all RNGs used by Uniform, E2LSH, and QuadTree (vendor copy).
  gen.seed(static_cast<unsigned>(args.seed));
  seed_quadtree_rng(static_cast<unsigned long>(args.seed));
  rng.seed(args.seed);
  std::cout << "seed=" << args.seed
            << (args.e2lsh_only ? " e2lsh_only=1" : "")
            << (args.quadtree_only ? " quadtree_only=1" : "")
            << (args.no_outlier ? " no_outlier=1" : "")
            << (args.dual_dynamic ? " dual_dynamic=1" : "")
            << " estimator=queryStratified"
            << (args.with_is ? " with_is=1" : "")
            << (args.T_override > 0 ? (" T_override=" + std::to_string(args.T_override)) : "")
            << (!args.storage_dir.empty() ? (" storage_dir=" + args.storage_dir) : "")
            << std::endl;
  auto wall_start = Clock::now();

  // f32bin A stays one float mapping. Exact-cache-only still wants nested A.
  // Outlier injection appends a row, so that path stays nested.
  FlatF32 flatA;
  bool use_flat_A = false;
  const bool have_outlier_file =
      file_exists_path(args.data_dir + "/base_with_outlier.f32bin") ||
      file_exists_path(args.data_dir + "/train_with_outlier.f32bin") ||
      file_exists_path(args.data_dir + "/base_with_outlier.fvecs") ||
      file_exists_path(args.data_dir + "/train_with_outlier.csv");
  const bool want_inject = !args.no_outlier && (args.inject_outlier || !have_outlier_file);
  const std::string a_f32 = args.no_outlier
                               ? (file_exists_path(args.data_dir + "/base.f32bin")
                                      ? args.data_dir + "/base.f32bin"
                                      : args.data_dir + "/train.f32bin")
                               : (file_exists_path(args.data_dir + "/base_with_outlier.f32bin")
                                      ? args.data_dir + "/base_with_outlier.f32bin"
                                      : args.data_dir + "/train_with_outlier.f32bin");
  if (!args.exact_cache_only && !want_inject && file_exists_path(a_f32)) {
    std::cout << "loading " << a_f32 << " (flat f32)\n";
    flatA = read_f32bin_flat(a_f32);
    use_flat_A = !flatA.empty();
  }

  std::vector<std::vector<double>> dataA, dataB;
  if (!use_flat_A) {
    if (args.no_outlier) {
      dataA = load_matrix_prefer(args.data_dir, {
          "train.f32bin",
          "base.f32bin",
          "train.csv",
          "base.fvecs",
          "sift_base.fvecs",
      });
    } else {
      dataA = load_matrix_prefer(args.data_dir, {
          "base_with_outlier.f32bin",
          "train_with_outlier.f32bin",
          "base_with_outlier.fvecs",
          "train_with_outlier.csv",
          "base.fvecs",
          "sift_base.fvecs",
          "train.csv",
      });
    }
  }
  dataB = load_matrix_prefer(args.data_dir, {
      "query.f32bin",
      "query.fvecs",
      "test.csv",
      "sift_query.fvecs",
  });
  if ((!use_flat_A && dataA.empty()) || dataB.empty()) {
    std::cerr << "Failed to read A/B from " << args.data_dir
              << " (need f32bin/csv pair; --no-outlier uses train.f32bin/base.f32bin)\n";
    return 1;
  }

  // If loaded raw base (no outlier file), allow --inject-outlier.
  if (args.no_outlier && args.inject_outlier) {
    std::cerr << "ERROR: --no-outlier and --inject-outlier are mutually exclusive\n";
    return 1;
  }
  if (!args.no_outlier && !use_flat_A && (args.inject_outlier || !have_outlier_file)) {
    if (!have_outlier_file) {
      std::cout << "no *outlier* artifact found; injecting outlier into A\n";
    }
    inject_sift_outlier(dataA);
  }
  if (use_flat_A && want_inject) {
    std::cerr << "ERROR: outlier injection not supported on the flat-A path; "
              << "use base_with_outlier.f32bin\n";
    return 1;
  }

  if (args.smoke) {
    if (use_flat_A) {
      if (static_cast<int>(flatA.n) > args.smoke_A) {
        flatA.n = static_cast<uint64_t>(args.smoke_A);
      }
    } else if (static_cast<int>(dataA.size()) > args.smoke_A) {
      dataA.resize(args.smoke_A);
    }
    if (static_cast<int>(dataB.size()) > args.smoke_B) dataB.resize(args.smoke_B);
    std::cout << "SMOKE mode: |A|=" << (use_flat_A ? flatA.n : dataA.size())
              << " |B|=" << dataB.size() << std::endl;
  }
  if (args.max_A > 0) {
    if (use_flat_A) {
      if (static_cast<int>(flatA.n) > args.max_A) {
        flatA.n = static_cast<uint64_t>(args.max_A);
        std::cout << "truncated |A| to " << flatA.n << " (--max-A)\n";
      }
    } else if (static_cast<int>(dataA.size()) > args.max_A) {
      dataA.resize(static_cast<size_t>(args.max_A));
      std::cout << "truncated |A| to " << dataA.size() << " (--max-A)\n";
    }
  }
  if (args.max_B > 0 && static_cast<int>(dataB.size()) > args.max_B) {
    dataB.resize(static_cast<size_t>(args.max_B));
    std::cout << "truncate |B| to " << dataB.size() << " (--max-B)\n";
  }

  std::vector<std::vector<double>> A, B;
  if (use_flat_A) {
    B = std::move(dataB);
    // A stays empty; points streamed from flatA
  } else if (dataA.size() >= dataB.size()) {
    A = std::move(dataA);
    B = std::move(dataB);
  } else {
    A = std::move(dataB);
    B = std::move(dataA);
  }

  const size_t nA = use_flat_A ? static_cast<size_t>(flatA.n) : A.size();
  const int dim = use_flat_A ? static_cast<int>(flatA.d) : static_cast<int>(A[0].size());
  if (dim != static_cast<int>(DIM)) {
    std::cerr << "DIM mismatch: data=" << dim << " config=" << DIM << std::endl;
    return 1;
  }

  int window_size = (args.window_size > 0)
                        ? args.window_size
                        : static_cast<int>(B.size()) / 20;
  int query_interval = (args.query_interval > 0)
                           ? args.query_interval
                           : std::max(static_cast<int>(B.size()) / 56, 1);
  if (window_size < 1) window_size = 1;
  if (query_interval < 1) query_interval = 1;
  std::cout << "window_size=" << window_size << " query_interval=" << query_interval
            << " |A|=" << nA << " |B|=" << B.size() << " dim=" << dim
            << (use_flat_A ? " (flat-A)" : "") << std::endl;

  if (args.dual_dynamic) {
    return run_dual_dynamic(args, A, B, dim, window_size, query_interval, wall_start,
                            use_flat_A ? &flatA : nullptr);
  }

  bool exact_from_cache = false;
  std::map<size_t, double> exact_cache;
  std::vector<std::pair<size_t, double>> exact_cache_to_write;
  if (!args.exact_cache_path.empty() && file_readable(args.exact_cache_path)) {
    exact_cache = load_exact_cache(args.exact_cache_path);
    exact_from_cache = true;
    std::cout << "exact=cache hit path=" << args.exact_cache_path
              << " entries=" << exact_cache.size() << std::endl;
  } else if (!args.exact_cache_path.empty()) {
    std::cout << "exact=computed, writing cache path=" << args.exact_cache_path << std::endl;
  } else {
    std::cout << "exact=computed (no --exact-cache)" << std::endl;
  }
  if (args.exact_cache_only && args.exact_cache_path.empty()) {
    std::cerr << "ERROR: --exact-cache-only requires --exact-cache <path>\n";
    return 1;
  }

  // --exact-cache-only: build/write exact cache without E2LSH / QuadTree.
  if (args.exact_cache_only) {
    if (use_flat_A) {
      std::cerr << "ERROR: --exact-cache-only needs nested A; omit --e2lsh-only or use non-f32bin\n";
      return 1;
    }
    if (exact_from_cache) {
      std::cout << "exact-cache-only: cache already present, nothing to do\n";
      return 0;
    }
    FastDynamicChamfer dc_only(A, {});
    std::deque<std::pair<std::vector<double>, char>> window;
    for (size_t i = 0; i < B.size(); ++i) {
      auto p = B[i];
      window.emplace_back(p, 'B');
      dc_only.insert_B(p);
      if (static_cast<int>(window.size()) > window_size) {
        dc_only.delete_B(0);
        window.pop_front();
      }
      if (static_cast<int>(i) >= window_size && i % query_interval == 0) {
        exact_cache_to_write.emplace_back(i, dc_only.current());
        std::cout << "exact-cache-only step " << i
                  << " chamfer=" << dc_only.current() << std::endl;
      }
    }
    write_exact_cache(args.exact_cache_path, args.data_dir, nA, B.size(),
                      window_size, query_interval, exact_cache_to_write);
    std::cout << "Wrote exact cache " << args.exact_cache_path
              << " entries=" << exact_cache_to_write.size() << std::endl;
    return 0;
  }

  // Wall-clock comparison. E2LSH queries and Benchmark updates are OpenMP;
  // QuadTree and Uniform are not. One thread makes the timers comparable.
  omp_set_num_threads(1);
  std::cout << "omp_threads=" << omp_get_max_threads() << "\n";

  if (args.e2lsh_only) {
    std::cout << "E2LSH-only mode: skipping QuadTree and Uniform\n";
  }

  // QuadTree setup (vendor protocol) — skipped in --e2lsh-only
  double qt_init_insert_A_ms = 0.0;
  std::unique_ptr<QuadTree> tree;
  if (!args.e2lsh_only) {
    const double max_dist = use_flat_A ? diameter_2approx_flat(flatA, B) : diameter_2approx_nested(A, B);
    double aspect_ratio = calculate_aspect_ratio(A, B, max_dist);
    int depth = calculate_quadtree_depth(aspect_ratio);
    std::cout << "diameter=" << max_dist << " aspect=" << aspect_ratio << " depth=" << depth << std::endl;

    auto t_qt_init0 = Clock::now();
    tree = std::make_unique<QuadTree>(DIM, max_dist * 2, depth);
    if (use_flat_A) {
      CoordAccess acc;
      acc.Af = flatA.data;
      acc.nA = static_cast<int>(flatA.n);
      acc.Bn = &B;
      acc.nB = static_cast<int>(B.size());
      acc.dim = dim;
      tree->bind(acc);
      tree->build_from_bound();
    } else {
      std::vector<std::vector<double>> A_empty, B_empty;
      tree->build(A, B, A_empty, B_empty);
    }
    tree->Build_Tree_Sampler();
    for (int ai = 0; ai < static_cast<int>(nA); ++ai) {
      if (use_flat_A) tree->insert_id('A', ai);
      else tree->insert(A[static_cast<size_t>(ai)], 'A', ai, A, B);
    }
    qt_init_insert_A_ms = Duration(Clock::now() - t_qt_init0).count();
    std::cout << "QuadTree A insert done in " << qt_init_insert_A_ms << " ms\n";
  }

  // Native E2LSH (skipped under --quadtree-only to fit 1M SIFT in ~15GB RAM)
  const double rho = 1e-4;
  const int n_max = args.quadtree_only
                        ? 8
                        : ((args.n_max > 0)
                               ? args.n_max
                               : std::max(70000, static_cast<int>(nA) + window_size + 1024));
  if (!args.quadtree_only && static_cast<int>(nA) > n_max) {
    std::cerr << "ERROR: |A|=" << nA << " exceeds n_max=" << n_max
              << "; pass --n-max >= |A|\n";
    return 1;
  }
  // E2LSH keeps its own float copy of A (about 3.5 GiB on GIST-1M). A second
  // full exact index still does not fit beside the hash tables, so the flat
  // path needs a cache hit.
  if (use_flat_A && !args.quadtree_only && !exact_from_cache && !args.exact_cache_only) {
    std::cerr << "ERROR: flat-A E2LSH requires --exact-cache hit (or nested A load)\n";
    return 1;
  }
  E2 e2;
  if (!args.storage_dir.empty()) {
    e2.ext = std::make_unique<e2lsh::ExternalChamferEstimator>(dim, rho, n_max, args.seed,
                                                              args.storage_dir);
  } else {
    e2.ram = std::make_unique<e2lsh::ChamferEstimator>(dim, rho, n_max, args.seed);
  }
  double e2_init_insert_A_ms = 0.0;
  if (!args.quadtree_only) {
    std::cout << "E2LSH K=" << e2.K() << " rho=" << rho << " n_max=" << n_max
              << " (formula: K=max(1,ceil(4*ln(8*n_max^2))))"
              << (e2.is_ext() ? " backend=external" : " backend=ram") << std::endl;
    if (e2.is_ext()) {
      std::cout << "storage_dir=" << args.storage_dir
                << " O_DIRECT=" << e2.ext->storageDirect()
                << " async=" << e2.ext->storageIoUring() << std::endl;
    }
    auto t_e2_init0 = Clock::now();
    auto preexpand_flat = [&](size_t npts) {
      if (!e2.is_ext() || npts == 0) return;
      std::vector<double> row(static_cast<size_t>(dim));
      const float* r0 = flatA.row(0);
      for (int j = 0; j < dim; ++j) row[static_cast<size_t>(j)] = static_cast<double>(r0[j]);
      e2.expandToCover(row);
      double R = 0.0;
      for (size_t ai = 0; ai < npts; ++ai) {
        const float* src = flatA.row(ai);
        double s = 0.0;
        for (int j = 0; j < dim; ++j) {
          double d = static_cast<double>(src[j]) - row[static_cast<size_t>(j)];
          s += d * d;
        }
        R = std::max(R, std::sqrt(s));
      }
      e2.expandToCoverR(R);
    };
    auto preexpand_nested = [&](const std::vector<std::vector<double>>& pts) {
      if (!e2.is_ext() || pts.empty()) return;
      e2.expandToCover(pts[0]);
      double R = 0.0;
      for (const auto& p : pts) R = std::max(R, e2lsh::l2(p, pts[0]));
      e2.expandToCoverR(R);
    };
    // For large SIFT: stream from flat f32 or free each nested row after insert.
    if (use_flat_A) {
      preexpand_flat(nA);
      const size_t chunk_n = 16384;
      std::vector<std::vector<double>> chunk;
      chunk.reserve(chunk_n);
      for (size_t ai = 0; ai < nA; ) {
        const size_t n = std::min(chunk_n, nA - ai);
        chunk.resize(n);
        for (size_t k = 0; k < n; ++k) {
          if (chunk[k].size() != static_cast<size_t>(dim))
            chunk[k].resize(static_cast<size_t>(dim));
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
        for (int k = 0; k < static_cast<int>(n); ++k) {
          const float* src = flatA.row(ai + static_cast<size_t>(k));
          double* dst = chunk[static_cast<size_t>(k)].data();
          for (int j = 0; j < dim; ++j) dst[j] = static_cast<double>(src[j]);
        }
#else
        for (size_t k = 0; k < n; ++k) {
          const float* src = flatA.row(ai + k);
          for (int j = 0; j < dim; ++j)
            chunk[k][static_cast<size_t>(j)] = static_cast<double>(src[j]);
        }
#endif
        (void)e2.insertABatch(chunk);
        ai += n;
        if (ai % 100000 < n || ai == nA) {
          std::cout << "E2LSH insertA progress " << ai << "/" << nA << std::endl;
        }
      }
      if (args.e2lsh_only) flatA.reset();  // QuadTree still reads this mapping
    } else if (args.e2lsh_only && exact_from_cache) {
      preexpand_nested(A);
      for (size_t ai = 0; ai < A.size(); ++ai) {
        (void)e2.insertA(A[ai]);
        A[ai].clear();
        A[ai].shrink_to_fit();
        if ((ai + 1) % 100000 == 0) {
          std::cout << "E2LSH insertA progress " << (ai + 1) << "/" << A.size() << std::endl;
        }
      }
      A.clear();
      A.shrink_to_fit();
    } else {
      preexpand_nested(A);
      std::vector<e2lsh::PointId> a_ids = e2.insertABatch(A);
      (void)a_ids;
    }
    e2.finalizeA();
    e2_init_insert_A_ms = Duration(Clock::now() - t_e2_init0).count();
    std::cout << "E2LSH A insert done in " << e2_init_insert_A_ms
              << " ms L=" << e2.L() << " D=" << e2.totalD() << std::endl;
  } else {
    std::cout << "QuadTree-only mode: skipping E2LSH insertA\n";
  }

  // Exact dynamic Chamfer (skipped entirely on cache hit)
  std::unique_ptr<FastDynamicChamfer> dc;
  if (!exact_from_cache) {
    if (use_flat_A) {
      dc = std::make_unique<FastDynamicChamfer>(static_cast<size_t>(dim));
      std::vector<double> row(static_cast<size_t>(dim));
      for (size_t ai = 0; ai < nA; ++ai) {
        promote_row(flatA.row(ai), dim, row);
        (void)dc->insert_A(row);
      }
    } else {
      dc = std::make_unique<FastDynamicChamfer>(A, std::vector<std::vector<double>>{});
    }
  }

  std::deque<std::pair<std::vector<double>, char>> window;
  std::deque<std::vector<double>> current_B;
  std::deque<e2lsh::PointId> e2_B_ids;

  std::ofstream cost_log(args.out_csv);
  if (!cost_log.is_open()) {
    std::cerr << "Cannot open " << args.out_csv << std::endl;
    return 1;
  }
  if (args.quadtree_only) {
    cost_log << "step,chamfer,quadtree_est,uniform_est,quadtree_err,uniform_err\n";
  } else if (args.e2lsh_only && args.with_is) {
    cost_log << "step,chamfer,strat_est,is_est,strat_err,is_err\n";
  } else if (args.e2lsh_only) {
    cost_log << "step,chamfer,strat_est,strat_err\n";
  } else {
    cost_log << "step,chamfer,quadtree_est,e2lsh_est,uniform_est,quadtree_err,e2lsh_err,uniform_err\n";
  }

  double qt_update_ms_total = 0.0;
  double e2_update_ms_total = 0.0;
  double qt_query_ms_total = 0.0;
  double strat_query_ms_total = 0.0;
  double is_query_ms_total = 0.0;
  double uni_query_ms_total = 0.0;
  double exact_query_ms_total = 0.0;
  double exact_update_ms_total = 0.0;
  int update_ops = 0;  // each B insert (+ optional remove) counts as one sliding-window step
  int queries = 0;
  bool nn_checked = false;
  NnMicro nn_micro;

  for (size_t i = 0; i < B.size(); ++i) {
    auto p = B[i];
    window.emplace_back(p, 'B');

    // QuadTree insert/remove B (update only) — skipped in --e2lsh-only
    if (!args.e2lsh_only) {
      auto t0 = Clock::now();
      tree->insert(p, 'B', static_cast<int>(i) + B_ID_shift, A, B);
      if (static_cast<int>(window.size()) > window_size) {
        tree->remove(window.front().first, window.front().second, A, B);
      }
      qt_update_ms_total += Duration(Clock::now() - t0).count();
    }

    // E2LSH insert/remove B (update only) — skipped in --quadtree-only
    if (!args.quadtree_only) {
      auto t0 = Clock::now();
      e2lsh::PointId bid = e2.insertB(p);
      e2_B_ids.push_back(bid);
      if (static_cast<int>(window.size()) > window_size) {
        e2.deleteB(e2_B_ids.front());
        e2_B_ids.pop_front();
      }
      e2_update_ms_total += Duration(Clock::now() - t0).count();
    }

    // Exact update (Benchmark) — skipped on cache hit
    if (dc) {
      auto t0 = Clock::now();
      dc->insert_B(p);
      if (static_cast<int>(window.size()) > window_size) {
        dc->delete_B(0);
      }
      exact_update_ms_total += Duration(Clock::now() - t0).count();
    }

    ++update_ops;

    current_B.push_back(p);
    if (static_cast<int>(window.size()) > window_size) {
      if (window.front().second == 'B') current_B.pop_front();
      window.pop_front();
    }

    if (static_cast<int>(i) >= window_size && i % query_interval == 0) {
      int n = static_cast<int>(nA);
      int T = (args.T_override > 0)
                  ? args.T_override
                  : e2lsh::ChamferEstimator::computeDefaultT(static_cast<size_t>(n));
      // Strata invariant on first query for any stratified path (production + compare).
      if (!args.quadtree_only && queries == 0) {
        if (!e2.validateStrataInvariant()) {
          std::cerr << "FATAL: strata invariant sumN=" << e2.sumStratumN()
                    << " pos=" << e2.countPositiveDistanceA() << " at first query\n";
          return 2;
        }
        std::cout << "strata invariant OK sumN=" << e2.sumStratumN()
                  << " pos=" << e2.countPositiveDistanceA() << std::endl;
      }
      std::cout << "Step " << i << " |A|=" << n << " |Bwin|=" << current_B.size()
                << " T=" << T;
      if (!args.quadtree_only) std::cout << " L=" << e2.L();
      std::cout << std::endl;

      // Harness copy of the live window. Not charged to any method.
      AlignedDoubles Bflat;
      if (!args.e2lsh_only) {
        Bflat.reset(current_B.size() * static_cast<size_t>(dim));
        std::size_t off = 0;
        for (const auto& b : current_B) {
          std::memcpy(Bflat.data() + off, b.data(), sizeof(double) * static_cast<size_t>(dim));
          off += static_cast<size_t>(dim);
        }
      }
      if (!nn_checked && !args.e2lsh_only && !args.quadtree_only && e2.ram &&
          (use_flat_A ? !flatA.empty() : !A.empty())) {
        nn_checked = true;
        std::deque<size_t> probe;
        std::vector<std::vector<double>> probeA;
        const size_t nsrc = use_flat_A ? static_cast<size_t>(flatA.n) : A.size();
        const size_t nprobe = std::min(nsrc, static_cast<size_t>(64));
        if (use_flat_A) {
          probeA.resize(nprobe);
          for (size_t k = 0; k < nprobe; ++k) {
            promote_row(flatA.row(k), dim, probeA[k]);
            probe.push_back(k);
          }
        } else {
          for (size_t k = 0; k < nprobe; ++k) probe.push_back(k);
        }
        nn_micro = nn_microbench(*e2.ram, use_flat_A ? probeA : A, probe, Bflat.data(), dim, current_B.size());
        if (std::strcmp(nn_micro.status, "FAIL") == 0) {
          std::cerr << "FATAL: exact-NN kernels disagree; refusing an unfair timing run\n";
          return 3;
        }
      }

      double e2_est = 0.0, chamfer = 0.0;
      double qt_est = 0.0, uni_est = 0.0, strat_est = 0.0, is_est = 0.0;

      if (!args.quadtree_only) {
        auto t0 = Clock::now();
        strat_est = e2.queryStratified(T);
        strat_query_ms_total += Duration(Clock::now() - t0).count();
        e2_est = strat_est;  // full-protocol CSV still labels column e2lsh_est
      }
      if (args.with_is) {
        // Importance sampling draws from the same RNG. Rewind afterwards so the
        // next queryStratified sees the same stream as a stratified-only run.
        const auto saved_rng = e2.rngSnapshot();
        auto t0 = Clock::now();
        is_est = e2.queryImportance(T);
        is_query_ms_total += Duration(Clock::now() - t0).count();
        e2.rngRestore(saved_rng);
      }
      {
        auto t0 = Clock::now();
        if (exact_from_cache) {
          auto it = exact_cache.find(i);
          if (it == exact_cache.end()) {
            std::cerr << "FATAL: exact cache missing step " << i
                      << " path=" << args.exact_cache_path << std::endl;
            return 2;
          }
          chamfer = it->second;
        } else {
          chamfer = dc->current();
          if (!args.exact_cache_path.empty()) {
            exact_cache_to_write.emplace_back(i, chamfer);
          }
        }
        exact_query_ms_total += Duration(Clock::now() - t0).count();
      }

      if (!args.e2lsh_only) {
        {
          auto t0 = Clock::now();
          qt_est = sample_quadtree_flat(*tree, A, T, Bflat.data(), dim, current_B.size());
          qt_query_ms_total += Duration(Clock::now() - t0).count();
        }
        {
          auto t0 = Clock::now();
          uni_est = use_flat_A ? sample_uniform_f32(flatA, T, Bflat.data(), dim, current_B.size())
                              : sample_uniform_flat(A, T, Bflat.data(), dim, current_B.size());
          uni_query_ms_total += Duration(Clock::now() - t0).count();
        }
      }

      double e2_err = (e2_est - chamfer) / chamfer;
      double strat_err = (strat_est - chamfer) / chamfer;

      if (args.quadtree_only) {
        double qt_err = (qt_est - chamfer) / chamfer;
        double uni_err = (uni_est - chamfer) / chamfer;
        std::cout << "  chamfer=" << chamfer
                  << " qt=" << qt_est << " (" << qt_err << ")"
                  << " uni=" << uni_est << " (" << uni_err << ")\n";
        cost_log << i << "," << chamfer << "," << qt_est << "," << uni_est
                 << "," << qt_err << "," << uni_err << "\n";
      } else if (args.e2lsh_only && args.with_is) {
        const double is_err = (is_est - chamfer) / chamfer;
        std::cout << "  chamfer=" << chamfer
                  << " strat=" << strat_est << " (" << strat_err << ")"
                  << " is=" << is_est << " (" << is_err << ")\n";
        cost_log << i << "," << chamfer << "," << strat_est << "," << is_est
                 << "," << strat_err << "," << is_err << "\n";
      } else if (args.e2lsh_only) {
        std::cout << "  chamfer=" << chamfer
                  << " strat=" << strat_est << " (" << strat_err << ")\n";
        cost_log << i << "," << chamfer << "," << strat_est << "," << strat_err << "\n";
      } else {
        double qt_err = (qt_est - chamfer) / chamfer;
        double uni_err = (uni_est - chamfer) / chamfer;
        std::cout << "  chamfer=" << chamfer
                  << " qt=" << qt_est << " (" << qt_err << ")"
                  << " e2(strat)=" << e2_est << " (" << e2_err << ")"
                  << " uni=" << uni_est << " (" << uni_err << ")\n";
        cost_log << i << "," << chamfer << "," << qt_est << "," << e2_est << "," << uni_est
                 << "," << qt_err << "," << e2_err << "," << uni_err << "\n";
      }
      cost_log.flush();
      ++queries;
    }
  }

  cost_log.close();

  if (!exact_from_cache && !args.exact_cache_path.empty()) {
    write_exact_cache(args.exact_cache_path, args.data_dir, nA, B.size(),
                      window_size, query_interval, exact_cache_to_write);
    std::cout << "Wrote exact cache " << args.exact_cache_path
              << " entries=" << exact_cache_to_write.size() << std::endl;
  }

  double wall_ms_total = Duration(Clock::now() - wall_start).count();
  double qt_update_ms_avg = update_ops > 0 ? qt_update_ms_total / update_ops : 0.0;
  double e2_update_ms_avg = update_ops > 0 ? e2_update_ms_total / update_ops : 0.0;
  double qt_query_ms_avg = queries > 0 ? qt_query_ms_total / queries : 0.0;
  double strat_query_ms_avg = queries > 0 ? strat_query_ms_total / queries : 0.0;
  double is_query_ms_avg = queries > 0 ? is_query_ms_total / queries : 0.0;
  double uni_query_ms_avg = queries > 0 ? uni_query_ms_total / queries : 0.0;
  double exact_query_ms_avg = queries > 0 ? exact_query_ms_total / queries : 0.0;
  const double inv_ops = update_ops > 0 ? 1.0 / static_cast<double>(update_ops) : 0.0;
  // Static-A insertion stays in init_insert_A_ms and is not part of a window step.
  const double e2_per_update_ms = e2_update_ms_avg + strat_query_ms_total * inv_ops;
  const double qt_per_update_ms = qt_update_ms_avg + qt_query_ms_total * inv_ops;
  const double uni_per_update_ms = uni_query_ms_total * inv_ops;
  const double bench_per_update_ms =
      (update_ops > 0 ? exact_update_ms_total / static_cast<double>(update_ops) : 0.0) +
      exact_query_ms_total * inv_ops;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\n========== TIMING SUMMARY (ms) ==========\n";
  std::cout << "E2LSH K=" << e2.K() << " L_final=" << e2.L()
            << " queries=" << queries
            << " update_ops=" << update_ops
            << (args.e2lsh_only ? " mode=e2lsh-only" : (args.quadtree_only ? " mode=quadtree-only" : "")) << "\n";
  std::cout << "wall_ms_total=" << wall_ms_total << "\n";
  std::cout << "per_update_ms e2lsh=" << e2_per_update_ms << " quadtree=" << qt_per_update_ms
            << " uniform=" << uni_per_update_ms << " benchmark=" << bench_per_update_ms << "\n";
  if (!args.e2lsh_only) {
    std::cout << "--- QuadTree ---\n";
    std::cout << "  init_insert_A_ms=" << qt_init_insert_A_ms << "\n";
    std::cout << "  update_ms_total=" << qt_update_ms_total
              << "  update_ms_avg=" << qt_update_ms_avg << "\n";
    std::cout << "  query_ms_total=" << qt_query_ms_total
              << "  query_ms_avg=" << qt_query_ms_avg << "\n";
  }
  std::cout << "--- E2LSH structure ---\n";
  std::cout << "  init_insert_A_ms=" << e2_init_insert_A_ms << "\n";
  std::cout << "  update_ms_total=" << e2_update_ms_total
            << "  update_ms_avg=" << e2_update_ms_avg << "\n";
  std::cout << "--- E2LSH stratified ---\n";
  std::cout << "  query_ms_total=" << strat_query_ms_total
            << "  query_ms_avg=" << strat_query_ms_avg << "\n";
  if (args.with_is) {
    std::cout << "--- E2LSH importance ---\n";
    std::cout << "  query_ms_total=" << is_query_ms_total
              << "  query_ms_avg=" << is_query_ms_avg << "\n";
  }
  std::cout << "--- Optional ---\n";
  if (!args.e2lsh_only) {
    std::cout << "  uniform_query_ms_total=" << uni_query_ms_total
              << "  uniform_query_ms_avg=" << uni_query_ms_avg << "\n";
  }
  std::cout << "  exact_chamfer mode=" << (exact_from_cache ? "cache" : "computed")
            << " update_ms_total=" << exact_update_ms_total
            << " update_ms_avg=" << (update_ops > 0 ? exact_update_ms_total / update_ops : 0.0)
            << " query_ms_total=" << exact_query_ms_total
            << " query_ms_avg=" << exact_query_ms_avg;
  if (!args.exact_cache_path.empty()) {
    std::cout << " path=" << args.exact_cache_path;
  }
  std::cout << "\n";
  std::cout << "=========================================\n";
  std::cout << "Done. queries=" << queries << " wrote " << args.out_csv << std::endl;

  // Write timing_summary.json
  std::ofstream tj(args.timing_json);
  if (!tj.is_open()) {
    std::cerr << "Cannot open " << args.timing_json << std::endl;
    return 1;
  }
  tj << std::fixed << std::setprecision(6);
  tj << "{\n";
  tj << "  \"seed\": " << args.seed << ",\n";
  tj << "  \"K\": " << e2.K() << ",\n";
  tj << "  \"L_final\": " << e2.L() << ",\n";
  tj << "  \"rho\": " << rho << ",\n";
  tj << "  \"n_max\": " << n_max << ",\n";
  tj << "  \"e2lsh_only\": " << (args.e2lsh_only ? "true" : "false") << ",\n";
  tj << "  \"queries\": " << queries << ",\n";
  tj << "  \"T\": " << (args.T_override > 0 ? args.T_override : -1) << ",\n";
  tj << "  \"window_size\": " << window_size << ",\n";
  tj << "  \"query_interval\": " << query_interval << ",\n";
  tj << "  \"nA\": " << nA << ",\n";
  tj << "  \"nB\": " << B.size() << ",\n";
  tj << "  \"dim\": " << dim << ",\n";
  tj << "  \"update_ops\": " << update_ops << ",\n";
  tj << "  \"wall_ms_total\": " << wall_ms_total << ",\n";
  tj << "  \"omp_threads\": " << omp_get_max_threads() << ",\n";
  tj << "  \"nn_kernel\": \"simd_l2::min_l2_flat\",\n";
  tj << "  \"quadtree_only\": " << (args.quadtree_only ? "true" : "false") << ",\n";
  if (!args.e2lsh_only) {
    tj << "  \"quadtree\": {\n";
    tj << "    \"init_insert_A_ms\": " << qt_init_insert_A_ms << ",\n";
    tj << "    \"update_ms_total\": " << qt_update_ms_total << ",\n";
    tj << "    \"update_ms_avg\": " << qt_update_ms_avg << ",\n";
    tj << "    \"query_ms_total\": " << qt_query_ms_total << ",\n";
    tj << "    \"query_ms_avg\": " << qt_query_ms_avg << "\n";
    tj << "  },\n";
  }
  tj << "  \"storage_dir\": \"" << json_escape(args.storage_dir) << "\",\n";
  tj << "  \"storage_backend\": \"" << (e2.is_ext() ? "external" : "ram") << "\",\n";
  tj << "  \"primary_estimator\": \"queryStratified\",\n";
  tj << "  \"e2lsh\": {\n";
  tj << "    \"init_insert_A_ms\": " << e2_init_insert_A_ms << ",\n";
  tj << "    \"update_ms_total\": " << e2_update_ms_total << ",\n";
  tj << "    \"update_ms_avg\": " << e2_update_ms_avg << ",\n";
  tj << "    \"query_ms_total\": " << strat_query_ms_total << ",\n";
  tj << "    \"query_ms_avg\": " << strat_query_ms_avg << "\n";
  tj << "  },\n";
  tj << "  \"e2lsh_stratified\": {\n";
  tj << "    \"query_ms_total\": " << strat_query_ms_total << ",\n";
  tj << "    \"query_ms_avg\": " << strat_query_ms_avg << ",\n";
  tj << "    \"fallback_weight\": \"N_fb * 2 * Delta_L (docs)\"\n";
  tj << "  },\n";
  tj << "  \"with_is\": " << (args.with_is ? "true" : "false") << ",\n";
  if (args.with_is) {
    tj << "  \"e2lsh_is\": {\n";
    tj << "    \"query_ms_total\": " << is_query_ms_total << ",\n";
    tj << "    \"query_ms_avg\": " << is_query_ms_avg << "\n";
    tj << "  },\n";
  }
  if (!args.e2lsh_only) {
    tj << "  \"uniform\": {\n";
    tj << "    \"query_ms_total\": " << uni_query_ms_total << ",\n";
    tj << "    \"query_ms_avg\": " << uni_query_ms_avg << "\n";
    tj << "  },\n";
  }
  tj << "  \"exact_chamfer\": {\n";
  tj << "    \"mode\": \"" << (exact_from_cache ? "cache" : "computed") << "\",\n";
  if (!args.exact_cache_path.empty()) {
    tj << "    \"cache_path\": \"" << json_escape(args.exact_cache_path) << "\",\n";
  }
  tj << "    \"update_ms_total\": " << exact_update_ms_total << ",\n";
  tj << "    \"update_ms_avg\": " << (update_ops > 0 ? exact_update_ms_total / static_cast<double>(update_ops) : 0.0) << ",\n";
  tj << "    \"query_ms_total\": " << exact_query_ms_total << ",\n";
  tj << "    \"query_ms_avg\": " << exact_query_ms_avg << "\n";
  tj << "  },\n";
  tj << "  \"per_update_ms\": {\n";
  {
    bool first_per = true;
    auto emit_per = [&](const char* key, double value) {
      if (!first_per) tj << ",\n";
      first_per = false;
      tj << "    \"" << key << "\": " << value;
    };
    if (!args.quadtree_only) emit_per("E2LSH", e2_per_update_ms);
    if (!args.e2lsh_only) {
      emit_per("QuadTree", qt_per_update_ms);
      emit_per("Uniform", uni_per_update_ms);
    }
    if (!exact_from_cache) emit_per("Benchmark", bench_per_update_ms);
    tj << "\n";
  }
  tj << "  },\n";
  tj << "  \"nn_microbench\": {\n";
  tj << "    \"reps\": " << nn_micro.reps << ",\n";
  tj << "    \"e2_ms\": " << nn_micro.e2_ms << ",\n";
  tj << "    \"flat_ms\": " << nn_micro.flat_ms << ",\n";
  tj << "    \"ratio_slow_over_fast\": " << nn_micro.ratio << ",\n";
  tj << "    \"e2_nB\": " << nn_micro.e2_nB << ",\n";
  tj << "    \"flat_nB\": " << nn_micro.flat_nB << ",\n";
  tj << "    \"note\": \"" << nn_micro.note << "\",\n";
  tj << "    \"status\": \"" << nn_micro.status << "\"\n";
  tj << "  },\n";
  {
    const auto& p = e2.profile();
    auto avg = [](double total, int64_t n) -> double {
      return n > 0 ? total / static_cast<double>(n) : 0.0;
    };
    tj << "  \"e2lsh_breakdown\": {\n";
    tj << "    \"insertA\": {\n";
    tj << "      \"count\": " << p.insertA_count << ",\n";
    tj << "      \"h0_ms\": " << p.insertA_h0_ms << ",\n";
    tj << "      \"hash_ms\": " << p.insertA_hash_ms << ",\n";
    tj << "      \"exact_ms\": " << p.insertA_exact_ms << ",\n";
    tj << "      \"map_ms\": " << p.insertA_map_ms << ",\n";
    tj << "      \"coord_io_ms\": " << p.insertA_coord_io_ms << ",\n";
    tj << "      \"posting_ms\": " << p.insertA_posting_ms << ",\n";
    tj << "      \"resolveLayerPath_ms\": " << p.insertA_resolve_ms << ",\n";
    tj << "      \"sampler_ms\": " << p.insertA_sampler_ms << ",\n";
    tj << "      \"other_ms\": " << p.insertA_other_ms << ",\n";
    tj << "      \"total_ms\": " << p.insertA_total_ms() << ",\n";
    tj << "      \"avg_ms\": " << avg(p.insertA_total_ms(), p.insertA_count) << "\n";
    tj << "    },\n";
    tj << "    \"insertB\": {\n";
    tj << "      \"count\": " << p.insertB_count << ",\n";
    tj << "      \"h0_ms\": " << p.insertB_h0_ms << ",\n";
    tj << "      \"resolveLayerPath_ms\": " << p.insertB_resolve_ms << ",\n";
    tj << "      \"exact_sampler_ms\": " << p.insertB_exact_sampler_ms << ",\n";
    tj << "      \"other_ms\": " << p.insertB_other_ms << ",\n";
    tj << "      \"total_ms\": " << p.insertB_total_ms() << ",\n";
    tj << "      \"avg_ms\": " << avg(p.insertB_total_ms(), p.insertB_count) << "\n";
    tj << "    },\n";
    tj << "    \"deleteB\": {\n";
    tj << "      \"count\": " << p.deleteB_count << ",\n";
    tj << "      \"membership_ms\": " << p.deleteB_membership_ms << ",\n";
    tj << "      \"exact_sampler_ms\": " << p.deleteB_exact_sampler_ms << ",\n";
    tj << "      \"other_ms\": " << p.deleteB_other_ms << ",\n";
    tj << "      \"total_ms\": " << p.deleteB_total_ms() << ",\n";
    tj << "      \"avg_ms\": " << avg(p.deleteB_total_ms(), p.deleteB_count) << "\n";
    tj << "    },\n";
    tj << "    \"expandTop\": {\n";
    tj << "      \"count\": " << p.expandTop_count << ",\n";
    tj << "      \"total_ms\": " << p.expandTop_ms << ",\n";
    tj << "      \"avg_ms\": " << avg(p.expandTop_ms, p.expandTop_count) << "\n";
    tj << "    },\n";
    tj << "    \"query\": {\n";
    tj << "      \"calls\": " << p.query_calls << ",\n";
    tj << "      \"samples\": " << p.query_samples << ",\n";
    tj << "      \"sample_fenwick_ms\": " << p.query_sample_ms << ",\n";
    tj << "      \"recover_Da_ms\": " << p.query_da_ms << ",\n";
    tj << "      \"exact_NN_ms\": " << p.query_nn_ms << ",\n";
    tj << "      \"overhead_ms\": " << p.query_overhead_ms << ",\n";
    tj << "      \"total_ms\": " << p.query_total_ms() << ",\n";
    tj << "      \"avg_per_call_ms\": " << avg(p.query_total_ms(), p.query_calls) << ",\n";
    double qtot = p.query_total_ms();
    auto pct = [qtot](double x) -> double { return qtot > 0 ? 100.0 * x / qtot : 0.0; };
    tj << "      \"pct_sample\": " << pct(p.query_sample_ms) << ",\n";
    tj << "      \"pct_Da\": " << pct(p.query_da_ms) << ",\n";
    tj << "      \"pct_NN\": " << pct(p.query_nn_ms) << ",\n";
    tj << "      \"pct_overhead\": " << pct(p.query_overhead_ms) << ",\n";
    tj << "      \"io_ms\": " << p.query_io_ms << ",\n";
    tj << "      \"io_reads\": " << p.io_reads << ",\n";
    tj << "      \"io_writes\": " << p.io_writes << ",\n";
    tj << "      \"io_bytes_read\": " << p.io_bytes_read << "\n";
    tj << "    }\n";
    tj << "  }\n";
  }
  tj << "}\n";
  tj.close();
  std::cout << "Wrote timing summary to " << args.timing_json << std::endl;

  // Human-readable bottleneck summary
  {
    const auto& p = e2.profile();
    double qtot = p.query_total_ms();
    double struct_ms = p.insertA_total_ms() + p.insertB_total_ms() + p.deleteB_total_ms() + p.expandTop_ms;
    double e2_profiled = struct_ms + qtot;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n========== E2LSH BREAKDOWN (ms) ==========\n";
    std::cout << "insertA total=" << p.insertA_total_ms()
              << " (h0=" << p.insertA_h0_ms
              << " hash=" << p.insertA_hash_ms
              << " exact=" << p.insertA_exact_ms
              << " map=" << p.insertA_map_ms
              << " coord_io=" << p.insertA_coord_io_ms
              << " posting=" << p.insertA_posting_ms
              << " resolve=" << p.insertA_resolve_ms
              << " sampler=" << p.insertA_sampler_ms
              << " other=" << p.insertA_other_ms << ")\n";
    std::cout << "insertB total=" << p.insertB_total_ms()
              << " (h0=" << p.insertB_h0_ms
              << " resolve=" << p.insertB_resolve_ms
              << " exact+sampler=" << p.insertB_exact_sampler_ms
              << " other=" << p.insertB_other_ms << ")\n";
    std::cout << "deleteB total=" << p.deleteB_total_ms()
              << " (membership=" << p.deleteB_membership_ms
              << " exact+sampler=" << p.deleteB_exact_sampler_ms
              << " other=" << p.deleteB_other_ms << ")\n";
    std::cout << "expandTop total=" << p.expandTop_ms << " count=" << p.expandTop_count << "\n";
    std::cout << "query total=" << qtot
              << " (sample=" << p.query_sample_ms
              << " Da=" << p.query_da_ms
              << " NN=" << p.query_nn_ms
              << " overhead=" << p.query_overhead_ms
              << " io=" << p.query_io_ms
              << " reads=" << p.io_reads << ")\n";
    if (e2_profiled > 0) {
      std::cout << "fractions of profiled E2LSH (" << e2_profiled << " ms): "
                << "NN=" << (100.0 * p.query_nn_ms / e2_profiled) << "% "
                << "sample=" << (100.0 * p.query_sample_ms / e2_profiled) << "% "
                << "Da=" << (100.0 * p.query_da_ms / e2_profiled) << "% "
                << "structure=" << (100.0 * struct_ms / e2_profiled) << "%\n";
    }
    std::cout << "==========================================\n";
  }

  return 0;
}
