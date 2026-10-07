#pragma once
// Euclidean LSH building blocks: the projection count K, the default sample
// budget, finest-level bucket ids H_0 and their parents, and a 128-bit
// fingerprint dictionary.
// The shift is drawn at the finest layer, S_{0,j} ~ Unif[0, K ρ). Each coarser
// layer draws a Bernoulli bit ξ and uses floor((h + ξ) / 2). See fillH0 and
// parentKeyInPlace.
#include "simd_l2.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <mutex>
#include <utility>

namespace e2lsh {

using PointId = uint64_t;

struct Hash128 {
  uint64_t h1 = 0;
  uint64_t h2 = 0;
};

inline uint64_t hash128MulXor(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
  const __uint128_t r = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
  return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
#else
  const uint64_t a_lo = static_cast<uint32_t>(a);
  const uint64_t a_hi = a >> 32;
  const uint64_t b_lo = static_cast<uint32_t>(b);
  const uint64_t b_hi = b >> 32;
  const uint64_t p0 = a_lo * b_lo;
  const uint64_t p1 = a_lo * b_hi;
  const uint64_t p2 = a_hi * b_lo;
  const uint64_t p3 = a_hi * b_hi;
  const uint64_t mid = (p0 >> 32) + static_cast<uint32_t>(p1) + static_cast<uint32_t>(p2);
  const uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
  const uint64_t lo = (p0 & 0xffffffffULL) | (mid << 32);
  return lo ^ hi;
#endif
}

inline uint64_t hash128Load64(const uint8_t* p) {
  uint64_t v = 0;
  std::memcpy(&v, p, 8);
  return v;
}

// 128-bit fingerprint of a byte string. Dictionary keys are these fingerprints.
// A collision treats two different keys as one, with probability about 2^{-128}.
inline Hash128 hash128(const void* data, std::size_t n, uint64_t seed = 0) {
  static constexpr uint64_t s0 = 0xa0761d6478bd642full;
  static constexpr uint64_t s1 = 0xe7037ed1a0b428dbull;
  static constexpr uint64_t s2 = 0x8ebc6af09c88c6e3ull;
  static constexpr uint64_t s3 = 0x589965cc75374cc3ull;
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint64_t a = seed ^ s0;
  uint64_t b = seed ^ s1;
  std::size_t i = 0;
  while (i + 16 <= n) {
    const uint64_t x = hash128Load64(p + i);
    const uint64_t y = hash128Load64(p + i + 8);
    a = hash128MulXor(x ^ s1, y ^ a);
    b = hash128MulXor(y ^ s0, x ^ b);
    i += 16;
  }
  uint8_t tail[16];
  std::memset(tail, 0, sizeof(tail));
  if (i < n) std::memcpy(tail, p + i, n - i);
  a = hash128MulXor(hash128Load64(tail) ^ s2, hash128Load64(tail + 8) ^ a);
  b = hash128MulXor(hash128Load64(tail + 8) ^ s3, hash128Load64(tail) ^ b);
  a = hash128MulXor(a, b ^ static_cast<uint64_t>(n) ^ s0);
  b = hash128MulXor(b, a ^ static_cast<uint64_t>(n) ^ s1);
  return Hash128{a, b};
}

// Open-addressed map. Lookup compares two 64-bit fingerprints.
// id == 0 is an empty slot. 0xFFFFFFFF is a tombstone; the probe chain stays intact.
// The load factor counts live entries plus tombstones. Counting only live entries
// fills the table with tombstones under sliding deletes, and insert loops.
struct FingerMap {
  static constexpr uint32_t kEmpty = 0;
  static constexpr uint32_t kTomb = 0xFFFFFFFFu;
  struct Slot {
    uint64_t h1 = 0;
    uint32_t id = 0;
  };
  std::vector<Slot> slots;
  uint32_t mask = 0;
  uint32_t n = 0;
  uint32_t tombs = 0;

  bool overloaded() const {
    const uint64_t used = static_cast<uint64_t>(n) + tombs;
    return slots.empty() || used * 10u >= static_cast<uint64_t>(slots.size()) * 7u;
  }

  void grow() {
    const uint32_t cap = slots.empty() ? 64u : static_cast<uint32_t>(slots.size()) * 2u;
    std::vector<Slot> neu(cap);
    const uint32_t m = cap - 1;
    for (const Slot& s : slots) {
      if (s.id == kEmpty || s.id == kTomb) continue;
      uint32_t i = static_cast<uint32_t>(s.h1) & m;
      while (neu[i].id != kEmpty) i = (i + 1) & m;
      neu[i] = s;
    }
    slots.swap(neu);
    mask = m;
    tombs = 0;
  }

  void insert(uint64_t h1, uint32_t id) {
    if (id == kEmpty || id == kTomb) throw std::invalid_argument("FingerMap id reserved");
    for (int attempt = 0; attempt < 2; ++attempt) {
      if (overloaded()) grow();
      uint32_t i = static_cast<uint32_t>(h1) & mask;
      bool have_tomb = false;
      uint32_t tomb_at = 0;
      const uint32_t cap = static_cast<uint32_t>(slots.size());
      for (uint32_t t = 0; t < cap; ++t) {
        const uint32_t cur = slots[i].id;
        if (cur == kEmpty) {
          const uint32_t dst = have_tomb ? tomb_at : i;
          if (have_tomb) --tombs;
          slots[dst] = Slot{h1, id};
          ++n;
          return;
        }
        if (cur == kTomb && !have_tomb) {
          have_tomb = true;
          tomb_at = i;
        }
        i = (i + 1) & mask;
      }
      // No empty slot (tombstones filled the table). Rehash and retry once.
      grow();
    }
    throw std::runtime_error("FingerMap::insert: no empty slot after rehash");
  }

  template <class GetH2>
  uint32_t find(uint64_t h1, uint64_t h2, GetH2 get_h2) const {
    if (slots.empty()) return 0;
    uint32_t i = static_cast<uint32_t>(h1) & mask;
    for (uint32_t t = 0; t < static_cast<uint32_t>(slots.size()); ++t) {
      const uint32_t id = slots[i].id;
      if (id == kEmpty) return 0;
      if (id != kTomb && slots[i].h1 == h1 && get_h2(id) == h2) return id;
      i = (i + 1) & mask;
    }
    return 0;
  }

  void erase(uint64_t h1, uint32_t id) {
    if (slots.empty() || id == kEmpty) return;
    uint32_t i = static_cast<uint32_t>(h1) & mask;
    for (uint32_t t = 0; t < static_cast<uint32_t>(slots.size()); ++t) {
      const uint32_t cur = slots[i].id;
      if (cur == kEmpty) return;
      if (cur == id && slots[i].h1 == h1) {
        slots[i].id = kTomb;
        --n;
        ++tombs;
        return;
      }
      i = (i + 1) & mask;
    }
  }
};

// 把 FingerMap 切成 64 片，插入时只锁自己那一片。批量建层时才用得上；单线程查询可以当普通字典读。
struct ShardedFingerMap {
  static constexpr int kShardBits = 6;
  static constexpr int kShards = 1 << kShardBits;
  struct Shard {
    FingerMap map;
    std::mutex mu;
  };
  std::unique_ptr<Shard[]> shards;

  ShardedFingerMap() : shards(std::make_unique<Shard[]>(kShards)) {}
  ShardedFingerMap(ShardedFingerMap&&) noexcept = default;
  ShardedFingerMap& operator=(ShardedFingerMap&&) noexcept = default;
  ShardedFingerMap(const ShardedFingerMap&) = delete;
  ShardedFingerMap& operator=(const ShardedFingerMap&) = delete;

  // High 6 bits: FingerMap still probes with h1 & mask (low bits). Using the same
  // low bits here packed each shard onto 1/64 of its slots.
  static int shardOf(uint64_t h1) { return static_cast<int>(h1 >> (64 - kShardBits)); }

  template <class GetH2>
  uint32_t find(uint64_t h1, uint64_t h2, GetH2 get_h2) const {
    return shards[static_cast<std::size_t>(shardOf(h1))].map.find(h1, h2, get_h2);
  }

  void insert(uint64_t h1, uint32_t id) { shards[static_cast<std::size_t>(shardOf(h1))].map.insert(h1, id); }

  void erase(uint64_t h1, uint32_t id) { shards[static_cast<std::size_t>(shardOf(h1))].map.erase(h1, id); }

  // Locked find-or-create. `create` must not take another shard lock.
  template <class GetH2, class Create>
  uint32_t findOrCreate(uint64_t h1, uint64_t h2, GetH2 get_h2, Create&& create) {
    Shard& sh = shards[static_cast<std::size_t>(shardOf(h1))];
    std::lock_guard<std::mutex> lock(sh.mu);
    uint32_t id = sh.map.find(h1, h2, get_h2);
    if (id) return id;
    id = create();
    sh.map.insert(h1, id);
    return id;
  }
};

// 数学上的向下取整除法 floor(a/b)，b>0。C++ 的整数除法朝 0 截断，负数桶号会算错。
inline int64_t floorDiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && a < 0) --q;
  return q;
}

// floor(a/2)。负的桶号不能靠「算术右移」，所以手写。
// 嵌套哈希往上走一层就要用它：父桶号 = floor( (当前桶号 + 随机 bit) / 2 )。
inline int64_t floorDiv2(int64_t a) {
  return a >= 0 ? a / 2 : (a - 1) / 2;
}

// K = ceil(4 ln(8 n^2)). n_max is the promised point bound.
inline int computeK(int n_max) {
  double v = 4.0 * std::log(8.0 * static_cast<double>(n_max) * static_cast<double>(n_max));
  int k = static_cast<int>(std::ceil(v));
  return std::max(1, k);
}

// Default query budget for one replica: T = floor(15 log2 |A|).
inline int defaultT(std::size_t a_size) {
  if (a_size == 0) return 1;
  return std::max(1, static_cast<int>(std::floor(15.0 * std::log2(static_cast<double>(a_size)))));
}

// 一个哈希桶的「门牌号」：K 个整数坐标 (h_1,...,h_K)，外加两枚指纹方便查字典。
// 层字典按 fingerprint 检索，插入热路径不必每次把冷桶的完整 K 元组从孩子那里拼回来。
struct BucketKey {
  int k = 0;
  std::vector<int64_t> h;
  std::size_t hash = 0;
  std::size_t hash2 = 0;

  void setK(int kk) {
    if (kk < 0) throw std::invalid_argument("BucketKey K out of range");
    k = kk;
    if (h.size() != static_cast<size_t>(kk)) h.assign(static_cast<size_t>(kk), 0);
  }

  void rehash() {
    const Hash128 hv = hash128(h.empty() ? nullptr : h.data(),
                               static_cast<std::size_t>(k) * sizeof(int64_t),
                               static_cast<uint64_t>(k));
    hash = static_cast<std::size_t>(hv.h1);
    hash2 = static_cast<std::size_t>(hv.h2);
  }

  // 完整 K 元组比较。层字典的查找路径并没有走这里，而是只比对两枚指纹。
  bool operator==(const BucketKey& o) const {
    if (k != o.k) return false;
    if (hash != o.hash || hash2 != o.hash2) return false;
    if (k == 0) return true;
    return std::memcmp(h.data(), o.h.data(), static_cast<size_t>(k) * sizeof(int64_t)) == 0;
  }
};

struct BucketKeyHash {
  std::size_t operator()(const BucketKey& key) const noexcept { return key.hash; }
};

// Fingerprint of one K-tuple.
inline void hashHRow(const int64_t* h, int K, uint64_t& hash1, uint64_t& hash2) {
  const Hash128 hv =
      hash128(h, static_cast<std::size_t>(K) * sizeof(int64_t), static_cast<uint64_t>(K));
  hash1 = hv.h1;
  hash2 = hv.h2;
}

// Finest-layer bucket. g_j ~ N(0, I_d), width w_0 = K ρ,
// h_{0,j} = floor((<g_j, x> + S0[j]) / w_0), with S0[j] ~ Unif[0, w_0).
// Coarser layers are not reprojected. Projection order matches simd_l2::dot.
inline void fillH0Row(const double* xd, int dim, int K, const double* g_flat, const double* S0,
                      double rho, int64_t* h, uint64_t& hash1, uint64_t& hash2) {
  const double w0 = static_cast<double>(K) * rho;
  for (int j = 0; j < K; ++j) {
    const double* gj = g_flat + static_cast<size_t>(j) * static_cast<size_t>(dim);
    const double proj = simd_l2::dot(gj, xd, dim);
    h[static_cast<size_t>(j)] =
        static_cast<int64_t>(std::floor((proj + S0[static_cast<size_t>(j)]) / w0));
  }
  hashHRow(h, K, hash1, hash2);
}

inline void fillH0(const double* xd, int dim, int K, const double* g_flat, const double* S0,
                   double rho, BucketKey& key) {
  key.setK(K);
  uint64_t h1 = 0, h2 = 0;
  fillH0Row(xd, dim, K, g_flat, S0, rho, key.h.data(), h1, h2);
  key.hash = static_cast<std::size_t>(h1);
  key.hash2 = static_cast<std::size_t>(h2);
}

// 批量 H_0：按点并行调用 fillH0Row，让 x_i（d 个 double）留在 L1 上做完 K 根投影。
// 与逐点 fillH0 bitwise 相同。row(i) 返回第 i 个点的 dim 个 double。
template <class Row>
inline void fillH0BatchRows(Row row, int n, int dim, int K, const double* g_flat, const double* S0,
                            double rho, int64_t* h_pack, uint64_t* hash1, uint64_t* hash2) {
  if (n <= 0) return;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n >= 64)
#endif
  for (int i = 0; i < n; ++i) {
    fillH0Row(row(i), dim, K, g_flat, S0, rho, h_pack + static_cast<size_t>(i) * static_cast<size_t>(K),
              hash1[static_cast<size_t>(i)], hash2[static_cast<size_t>(i)]);
  }
}

inline void fillH0Batch(const double* X, int n, int dim, int K, const double* g_flat,
                        const double* S0, double rho, int64_t* h_pack, uint64_t* hash1,
                        uint64_t* hash2) {
  fillH0BatchRows([&](int i) { return X + static_cast<size_t>(i) * static_cast<size_t>(dim); }, n, dim,
                  K, g_flat, S0, rho, h_pack, hash1, hash2);
}

// Parent bucket: H_{i+1,j} = floor((h_{i,j} + ξ_{i,j}) / 2).
// ξ is Bernoulli(1/2) and is stored in ChamferEstimator::xi_.
// Each fine bucket still has one coarse parent.
inline void parentKeyRowInPlace(int64_t* h, int K, const std::vector<uint8_t>& xi, uint64_t& hash1,
                                uint64_t& hash2) {
  for (int j = 0; j < K; ++j) {
    h[static_cast<size_t>(j)] =
        floorDiv2(h[static_cast<size_t>(j)] + static_cast<int64_t>(xi[static_cast<size_t>(j)]));
  }
  hashHRow(h, K, hash1, hash2);
}

inline void parentKeyInPlace(BucketKey& z, int K, const std::vector<uint8_t>& xi) {
  uint64_t h1 = static_cast<uint64_t>(z.hash);
  uint64_t h2 = static_cast<uint64_t>(z.hash2);
  parentKeyRowInPlace(z.h.data(), K, xi, h1, h2);
  z.hash = static_cast<std::size_t>(h1);
  z.hash2 = static_cast<std::size_t>(h2);
}

// Fingerprint of an exact coordinate. The dictionary stores (hash, hash2) only.
// A collision treats two different points as one cell, with probability about 2^{-128}.
inline void hashCoords(const double* p, int d, std::size_t& hash, std::size_t& hash2) {
  const Hash128 hv =
      hash128(p, static_cast<std::size_t>(d) * sizeof(double), static_cast<uint64_t>(d));
  hash = static_cast<std::size_t>(hv.h1);
  hash2 = static_cast<std::size_t>(hv.h2);
}

}  // namespace e2lsh
