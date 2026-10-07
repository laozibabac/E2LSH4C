#pragma once
// Dedicated squared-L2 / dot kernels (AVX-512 / AVX2 / scalar).
// Early-exit vs best bound is OFF by default; enable with -DE2LSH_NN_EARLY_EXIT=1.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#if defined(_WIN32)
#include <malloc.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#ifndef E2LSH_NN_EARLY_EXIT
#define E2LSH_NN_EARLY_EXIT 0
#endif

namespace e2lsh {
namespace simd_l2 {

inline double squared_l2_scalar(const double* a, const double* b, int dim) {
  double s = 0.0;
  for (int i = 0; i < dim; ++i) {
    double d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

#if defined(__AVX512F__)
inline double squared_l2_avx512(const double* a, const double* b, int dim) {
  __m512d acc = _mm512_setzero_pd();
  int i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m512d va = _mm512_loadu_pd(a + i);
    __m512d vb = _mm512_loadu_pd(b + i);
    __m512d d = _mm512_sub_pd(va, vb);
    acc = _mm512_fmadd_pd(d, d, acc);
  }
  double s = _mm512_reduce_add_pd(acc);
  for (; i < dim; ++i) {
    double d = a[i] - b[i];
    s += d * d;
  }
  return s;
}
#endif

#if defined(__AVX2__)
inline double squared_l2_avx2(const double* a, const double* b, int dim) {
  __m256d acc = _mm256_setzero_pd();
  int i = 0;
  for (; i + 4 <= dim; i += 4) {
    __m256d va = _mm256_loadu_pd(a + i);
    __m256d vb = _mm256_loadu_pd(b + i);
    __m256d d = _mm256_sub_pd(va, vb);
#if defined(__FMA__)
    acc = _mm256_fmadd_pd(d, d, acc);
#else
    acc = _mm256_add_pd(acc, _mm256_mul_pd(d, d));
#endif
  }
  alignas(32) double tmp[4];
  _mm256_store_pd(tmp, acc);
  double s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
  for (; i < dim; ++i) {
    double d = a[i] - b[i];
    s += d * d;
  }
  return s;
}
#endif

inline double squared_l2(const double* a, const double* b, int dim) {
#if defined(__AVX512F__)
  return squared_l2_avx512(a, b, dim);
#elif defined(__AVX2__)
  return squared_l2_avx2(a, b, dim);
#else
  return squared_l2_scalar(a, b, dim);
#endif
}


inline double dot_scalar(const double* a, const double* b, int dim) {
  double s = 0.0;
  for (int i = 0; i < dim; ++i) s += a[i] * b[i];
  return s;
}

#if defined(__AVX512F__)
inline double dot_avx512(const double* a, const double* b, int dim) {
  __m512d acc = _mm512_setzero_pd();
  int i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m512d va = _mm512_loadu_pd(a + i);
    __m512d vb = _mm512_loadu_pd(b + i);
    acc = _mm512_fmadd_pd(va, vb, acc);
  }
  double s = _mm512_reduce_add_pd(acc);
  for (; i < dim; ++i) s += a[i] * b[i];
  return s;
}
#endif

#if defined(__AVX2__)
inline double dot_avx2(const double* a, const double* b, int dim) {
  __m256d acc = _mm256_setzero_pd();
  int i = 0;
  for (; i + 4 <= dim; i += 4) {
    __m256d va = _mm256_loadu_pd(a + i);
    __m256d vb = _mm256_loadu_pd(b + i);
#if defined(__FMA__)
    acc = _mm256_fmadd_pd(va, vb, acc);
#else
    acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
#endif
  }
  alignas(32) double tmp[4];
  _mm256_store_pd(tmp, acc);
  double s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
  for (; i < dim; ++i) s += a[i] * b[i];
  return s;
}
#endif

inline double dot(const double* a, const double* b, int dim) {
#if defined(__AVX512F__)
  return dot_avx512(a, b, dim);
#elif defined(__AVX2__)
  return dot_avx2(a, b, dim);
#else
  return dot_scalar(a, b, dim);
#endif
}

// Min ℓ2 over contiguous B_flat layout [b0|b1|...|b_{nB-1}], each dim doubles.
inline double min_l2_flat(const double* q, int dim, const double* B_flat, std::size_t nB) {
  if (nB == 0) return std::numeric_limits<double>::infinity();
  double best_sq = std::numeric_limits<double>::infinity();
  const double* bp = B_flat;
  for (std::size_t j = 0; j < nB; ++j, bp += dim) {
#if E2LSH_NN_EARLY_EXIT
    // Ablation only: partial accumulate with early reject (default OFF).
    double s = 0.0;
    int i = 0;
    for (; i < dim; ++i) {
      double d = q[i] - bp[i];
      s += d * d;
      if (s >= best_sq) break;
    }
    if (i != dim) continue;
    if (s < best_sq) best_sq = s;
#else
    double s = squared_l2(q, bp, dim);
    if (s < best_sq) best_sq = s;
#endif
  }
  return std::sqrt(best_sq);
}

}  // namespace simd_l2

// 64-byte alignment. AVX-512 loads that split a cache line are about 2x slower
// at d=128, and the ordinary allocator only promises 16 bytes, so which seed
// gets the fast scan depends on heap layout.
template <class T, std::size_t Align = 64>
struct AlignedAlloc {
  static_assert(Align != 0 && (Align & (Align - 1)) == 0, "Align must be a power of two");
  using value_type = T;
  template <class U>
  struct rebind {
    using other = AlignedAlloc<U, Align>;
  };

  AlignedAlloc() noexcept = default;
  template <class U>
  AlignedAlloc(const AlignedAlloc<U, Align>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n > static_cast<std::size_t>(-1) / sizeof(T)) throw std::bad_alloc();
    const std::size_t bytes = n * sizeof(T);
    void* p = nullptr;
#if defined(_WIN32)
    p = _aligned_malloc(bytes == 0 ? Align : bytes, Align);
#else
    const std::size_t need = std::max(bytes, Align);
    const std::size_t rounded = (need + Align - 1) & ~(Align - 1);
    p = std::aligned_alloc(Align, rounded);
#endif
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }

  void deallocate(T* p, std::size_t) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
};

template <class T, class U, std::size_t Align>
inline bool operator==(const AlignedAlloc<T, Align>&, const AlignedAlloc<U, Align>&) noexcept {
  return true;
}
template <class T, class U, std::size_t Align>
inline bool operator!=(const AlignedAlloc<T, Align>&, const AlignedAlloc<U, Align>&) noexcept {
  return false;
}

}  // namespace e2lsh
