// Incremental exact one-sided Chamfer cache from f32bin A/B (mapped A, window B).
// Matches compare_main window / query_interval / cache CSV format.
// Also records Benchmark (exact) per-window-update time.
#include "os/mapped_file.hpp"
#include "simd_l2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

static void usage() {
  std::cerr << "usage: exact_cache_f32 --data-dir DIR --out PATH "
               "[--a-name FILE] [--window N] [--query-interval N]\n"
               "         [--timing-out PATH]\n"
               "         [--max-A N] [--max-B N]\n";
}

static double squared_l2_f32(const float* a, const float* b, int dim) {
#if defined(__AVX2__)
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 d = _mm256_sub_ps(va, vb);
#if defined(__FMA__)
    acc = _mm256_fmadd_ps(d, d, acc);
#else
    acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
#endif
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);
  double s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < dim; ++i) {
    double d = static_cast<double>(a[i] - b[i]);
    s += d * d;
  }
  return s;
#else
  double s = 0.0;
  for (int i = 0; i < dim; ++i) {
    double d = static_cast<double>(a[i] - b[i]);
    s += d * d;
  }
  return s;
#endif
}

int main(int argc, char** argv) {
  std::string data_dir, out_path, a_name = "base_with_outlier.f32bin";
  std::string timing_path;
  int window_size = -1, query_interval = -1, max_A = -1, max_B = -1;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
    else if (s == "--out" && i + 1 < argc) out_path = argv[++i];
    else if (s == "--a-name" && i + 1 < argc) a_name = argv[++i];
    else if ((s == "--window" || s == "--window-size") && i + 1 < argc)
      window_size = std::stoi(argv[++i]);
    else if ((s == "--query-interval" || s == "--query-every") && i + 1 < argc)
      query_interval = std::stoi(argv[++i]);
    else if (s == "--timing-out" && i + 1 < argc) timing_path = argv[++i];
    else if (s == "--max-A" && i + 1 < argc) max_A = std::stoi(argv[++i]);
    else if (s == "--max-B" && i + 1 < argc) max_B = std::stoi(argv[++i]);
    else {
      usage();
      return 2;
    }
  }
  if (data_dir.empty() || out_path.empty()) {
    usage();
    return 2;
  }
  omp_set_num_threads(1);
  std::cout << "omp_threads=" << omp_get_max_threads() << "\n";

  const std::string a_path = data_dir + "/" + a_name;
  const std::string b_path = data_dir + "/query.f32bin";

  e2lsh::os::MappedFile amap;
  if (!amap.openRead(a_path)) {
    std::cerr << "open A failed: " << a_path << "\n";
    return 1;
  }
  if (amap.size() < 16) {
    std::cerr << "fstat A failed\n";
    return 1;
  }
  const uint64_t nA = *reinterpret_cast<const uint64_t*>(amap.data());
  const uint64_t dA = *(reinterpret_cast<const uint64_t*>(amap.data()) + 1);
  const float* A = reinterpret_cast<const float*>(reinterpret_cast<const char*>(amap.data()) + 16);
  if (nA == 0 || dA == 0 || dA > 4096) {
    std::cerr << "bad A header n=" << nA << " d=" << dA << "\n";
    return 1;
  }

  std::ifstream bin(b_path, std::ios::binary);
  if (!bin) {
    std::cerr << "open B failed: " << b_path << "\n";
    return 1;
  }
  uint64_t nB = 0, dB = 0;
  bin.read(reinterpret_cast<char*>(&nB), 8);
  bin.read(reinterpret_cast<char*>(&dB), 8);
  if (dB != dA) {
    std::cerr << "dim mismatch A=" << dA << " B=" << dB << "\n";
    return 1;
  }
  std::vector<float> Bf(static_cast<size_t>(nB * dB));
  bin.read(reinterpret_cast<char*>(Bf.data()), static_cast<std::streamsize>(Bf.size() * 4));
  if (!bin) {
    std::cerr << "short B read\n";
    return 1;
  }

  uint64_t nA_use = nA;
  uint64_t nB_use = nB;
  if (max_A > 0 && static_cast<uint64_t>(max_A) < nA_use) nA_use = static_cast<uint64_t>(max_A);
  if (max_B > 0 && static_cast<uint64_t>(max_B) < nB_use) nB_use = static_cast<uint64_t>(max_B);

  if (window_size <= 0) window_size = static_cast<int>(nB_use) / 20;
  if (query_interval <= 0) query_interval = std::max(static_cast<int>(nB_use) / 56, 1);
  if (window_size < 1) window_size = 1;
  const int dim = static_cast<int>(dA);
  std::cout << "exact_cache_f32 |A|=" << nA_use << " |B|=" << nB_use << " dim=" << dim
            << " window=" << window_size << " query_interval=" << query_interval
            << " a=" << a_name << std::endl;

  const std::size_t nAi = static_cast<std::size_t>(nA_use);
  std::vector<double> min_sq(nAi, std::numeric_limits<double>::infinity());
  std::vector<std::uint64_t> nn_id(nAi, ~std::uint64_t{0});
  std::vector<double> min_dist(nAi, std::numeric_limits<double>::infinity());
  double sum = 0.0;

  struct WB {
    std::uint64_t id;
    std::vector<float> x;
  };
  std::deque<WB> win;
  std::uint64_t next_id = 0;
  std::vector<std::pair<size_t, double>> entries;
  entries.reserve(64);
  double update_ms_total = 0.0;
  double exact_query_ms_total = 0.0;
  int update_ops = 0;
  auto wall0 = Clock::now();

  for (size_t i = 0; i < static_cast<size_t>(nB_use); ++i) {
    WB nb;
    nb.id = next_id++;
    nb.x.resize(static_cast<size_t>(dim));
    const float* src = Bf.data() + i * static_cast<size_t>(dim);
    for (int d = 0; d < dim; ++d) nb.x[static_cast<size_t>(d)] = src[d];

    auto t0 = Clock::now();
    const float* bp = nb.x.data();
    double delta = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : delta)
#endif
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(nAi); ++ii) {
      const std::size_t a = static_cast<std::size_t>(ii);
      const float* ap = A + a * static_cast<size_t>(dim);
      const double sq = squared_l2_f32(ap, bp, dim);
      if (sq < min_sq[a]) {
        const double dist = std::sqrt(sq);
        const double old = min_dist[a];
        if (old == std::numeric_limits<double>::infinity())
          delta += dist;
        else
          delta += (dist - old);
        min_sq[a] = sq;
        min_dist[a] = dist;
        nn_id[a] = nb.id;
      }
    }
    sum += delta;
    win.push_back(std::move(nb));

    if (static_cast<int>(win.size()) > window_size) {
      const std::uint64_t old_id = win.front().id;
      win.pop_front();
      std::vector<std::size_t> need;
      need.reserve(nAi / 16 + 1);
      for (std::size_t a = 0; a < nAi; ++a) {
        if (nn_id[a] == old_id) {
          sum -= min_dist[a];
          min_dist[a] = std::numeric_limits<double>::infinity();
          min_sq[a] = std::numeric_limits<double>::infinity();
          nn_id[a] = ~std::uint64_t{0};
          need.push_back(a);
        }
      }
      if (!need.empty() && !win.empty()) {
        std::vector<const float*> bptr(win.size());
        std::vector<std::uint64_t> bid(win.size());
        for (std::size_t j = 0; j < win.size(); ++j) {
          bptr[j] = win[j].x.data();
          bid[j] = win[j].id;
        }
        const std::size_t nW = win.size();
        const std::size_t nR = need.size();
        double add = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : add)
#endif
        for (std::ptrdiff_t rr = 0; rr < static_cast<std::ptrdiff_t>(nR); ++rr) {
          const std::size_t a = need[static_cast<std::size_t>(rr)];
          const float* ap = A + a * static_cast<size_t>(dim);
          double best_sq = std::numeric_limits<double>::infinity();
          std::uint64_t best_id = ~std::uint64_t{0};
          for (std::size_t j = 0; j < nW; ++j) {
            const double sq = squared_l2_f32(ap, bptr[j], dim);
            if (sq < best_sq) {
              best_sq = sq;
              best_id = bid[j];
            }
          }
          const double dist = std::sqrt(best_sq);
          min_sq[a] = best_sq;
          min_dist[a] = dist;
          nn_id[a] = best_id;
          add += dist;
        }
        sum += add;
      }
    }
    const double step_ms = Duration(Clock::now() - t0).count();
    update_ms_total += step_ms;
    ++update_ops;

    if (static_cast<int>(i) >= window_size && i % static_cast<size_t>(query_interval) == 0) {
      auto tq = Clock::now();
      const double chamfer = sum;
      exact_query_ms_total += Duration(Clock::now() - tq).count();
      entries.emplace_back(i, chamfer);
      std::cout << "exact-cache step " << i << " chamfer=" << chamfer << std::endl;
    }
  }

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "cannot write " << out_path << "\n";
    return 1;
  }
  out << "# exact Chamfer cache\n";
  out << "# data_dir=" << data_dir << "\n";
  out << "# a_name=" << a_name << "\n";
  out << "# |A|=" << nA_use << "\n";
  out << "# |B|=" << nB_use << "\n";
  out << "# window_size=" << window_size << "\n";
  out << "# query_interval=" << query_interval << "\n";
  out << "step,chamfer\n";
  out << std::setprecision(17);
  for (const auto& e : entries) out << e.first << "," << e.second << "\n";
  const double wall_ms = Duration(Clock::now() - wall0).count();
  const double upd_avg = update_ops > 0 ? update_ms_total / update_ops : 0.0;
  const double inv_ops = update_ops > 0 ? 1.0 / static_cast<double>(update_ops) : 0.0;
  const double query_avg = entries.empty() ? 0.0 : exact_query_ms_total / static_cast<double>(entries.size());
  const double bench_per = upd_avg + exact_query_ms_total * inv_ops;
  std::cout << "Wrote " << out_path << " entries=" << entries.size()
            << " update_ms_total=" << update_ms_total
            << " update_ms_avg=" << upd_avg
            << " per_update_ms benchmark=" << bench_per
            << " wall_ms=" << wall_ms << std::endl;

  if (timing_path.empty()) timing_path = out_path + ".timing.json";
  std::ofstream tj(timing_path);
  if (tj) {
    tj << std::fixed << std::setprecision(6);
    tj << "{\n";
    tj << "  \"nA\": " << nA_use << ",\n";
    tj << "  \"nB\": " << nB_use << ",\n";
    tj << "  \"dim\": " << dim << ",\n";
    tj << "  \"window_size\": " << window_size << ",\n";
    tj << "  \"query_interval\": " << query_interval << ",\n";
    tj << "  \"a_name\": \"" << a_name << "\",\n";
    tj << "  \"omp_threads\": " << omp_get_max_threads() << ",\n";
    tj << "  \"nn_kernel\": \"squared_l2_f32\",\n";
    tj << "  \"update_ops\": " << update_ops << ",\n";
    tj << "  \"queries\": " << entries.size() << ",\n";
    tj << "  \"wall_ms_total\": " << wall_ms << ",\n";
    tj << "  \"exact_chamfer\": {\n";
    tj << "    \"mode\": \"computed\",\n";
    tj << "    \"update_ms_total\": " << update_ms_total << ",\n";
    tj << "    \"update_ms_avg\": " << upd_avg << ",\n";
    tj << "    \"query_ms_total\": " << exact_query_ms_total << ",\n";
    tj << "    \"query_ms_avg\": " << query_avg << "\n";
    tj << "  },\n";
    tj << "  \"per_update_ms\": {\n";
    tj << "    \"Benchmark\": " << bench_per << "\n";
    tj << "  }\n";
    tj << "}\n";
    std::cout << "Wrote " << timing_path << std::endl;
  }
  return 0;
}
