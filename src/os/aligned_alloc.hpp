#pragma once
// Sector-aligned allocation: posix_memalign / _aligned_malloc.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace e2lsh {
namespace os {

inline std::size_t alignUp(std::size_t n, std::size_t a) {
  if (a <= 1) return n;
  return (n + a - 1) / a * a;
}

inline std::size_t nextPow2(std::size_t a) {
  if (a <= 1) return 1;
  if ((a & (a - 1)) == 0) return a;
  std::size_t p = 1;
  while (p < a) p <<= 1;
  return p;
}

inline void* allocAlignedUninitialized(std::size_t bytes, std::size_t align) {
  align = nextPow2(std::max(align, sizeof(void*)));
  const std::size_t n = alignUp(std::max(bytes, align), align);
  void* p = nullptr;
#ifdef _WIN32
  p = _aligned_malloc(n, align);
  if (!p) throw std::runtime_error("_aligned_malloc failed");
#else
  if (posix_memalign(&p, align, n) != 0 || !p)
    throw std::runtime_error("posix_memalign failed");
#endif
  return p;
}

inline void* allocAligned(std::size_t bytes, std::size_t align) {
  align = nextPow2(std::max(align, sizeof(void*)));
  const std::size_t n = alignUp(std::max(bytes, align), align);
  void* p = allocAlignedUninitialized(bytes, align);
  std::memset(p, 0, n);
  return p;
}

inline void freeAligned(void* p) {
  if (!p) return;
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

// Growable sector-aligned bounce buffer. Contents are uninitialized (I/O scratch).
class AlignedScratch {
 public:
  AlignedScratch() = default;
  ~AlignedScratch() { reset(); }
  AlignedScratch(const AlignedScratch&) = delete;
  AlignedScratch& operator=(const AlignedScratch&) = delete;

  void reset() {
    if (p_) {
      freeAligned(p_);
      p_ = nullptr;
    }
    bytes_ = 0;
    align_ = 0;
  }

  void* ensure(std::size_t bytes, std::size_t align) {
    if (p_ && bytes_ >= bytes && align_ == align) return p_;
    const std::size_t cap = bytes_ == 0 ? bytes : std::max(bytes, bytes_ * 2);
    void* np = allocAlignedUninitialized(cap, align);
    if (p_) freeAligned(p_);
    p_ = np;
    bytes_ = cap;
    align_ = align;
    return p_;
  }

 private:
  void* p_ = nullptr;
  std::size_t bytes_ = 0;
  std::size_t align_ = 0;
};

}  // namespace os
}  // namespace e2lsh
