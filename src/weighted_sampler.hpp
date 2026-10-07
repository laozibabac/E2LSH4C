#pragma once
// Fenwick tree of floating weights. Each stratum owns one sampler over frontier
// components (an exact cell, or a hash bucket, whose B is empty). The weight is
// the component's A count. An update is O(log(n+1)). When that weight is an
// integer A count, the within-layer draw is still uniform.
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <random>
#include <cmath>
#include <limits>

// 采样器里一个槽位的「取货号」，setWeight / destroy 时凭它找到那一行。
struct SamplerSlotRef {
  int index = -1;
};

enum class ComponentKind : uint8_t { Exact = 0, Bucket = 1 };

// 抽到的那一条目是谁：精确坐标叶子，还是某层哈希桶。ptr 指向对应对象。
struct ComponentId {
  ComponentKind kind = ComponentKind::Exact;
  int layer = -1;   // 桶才有意义，告诉你这是第几层
  void* ptr = nullptr;
};

// 重量存在 Fenwick 里，create / setWeight / sample 都围着它转。容量按 2 的幂涨。
class DynamicWeightedSampler {
 public:
  DynamicWeightedSampler() { reserve(1024); }

  // 登记一条新条目，重量记进 Fenwick。返回的 ref 以后改重量、删除都靠它。
  SamplerSlotRef create(double weight, ComponentId id) {
    if (weight < 0.0) weight = 0.0;
    int idx;
    if (!free_list_.empty()) {
      idx = free_list_.back();
      free_list_.pop_back();
    } else {
      idx = live_count_;
      if (idx >= capacity_) grow();
      ++live_count_;
    }
    weights_[idx] = weight;
    ids_[idx] = id;
    alive_[idx] = true;
    fenwickAdd(idx + 1, weight);
    total_ += weight;
    return SamplerSlotRef{idx};
  }

  // O(log capacity) Fenwick add.
  void setWeight(SamplerSlotRef ref, double new_w) {
    const int idx = ref.index;
    if (idx < 0 || idx >= live_count_ || !alive_[idx]) return;
    if (new_w < 0.0) new_w = 0.0;
    const double delta = new_w - weights_[idx];
    if (delta == 0.0) return;
    weights_[idx] = new_w;
    fenwickAdd(idx + 1, delta);
    total_ += delta;
  }

  void addWeight(SamplerSlotRef ref, double delta) {
    const int idx = ref.index;
    if (idx < 0 || idx >= live_count_ || !alive_[idx]) return;
    double nw = weights_[idx] + delta;
    if (nw < 0.0) {
      delta = -weights_[idx];
      nw = 0.0;
    }
    weights_[idx] = nw;
    fenwickAdd(idx + 1, delta);
    total_ += delta;
  }

  void destroy(SamplerSlotRef ref) {
    const int idx = ref.index;
    if (idx < 0 || idx >= live_count_ || !alive_[idx]) return;
    fenwickAdd(idx + 1, -weights_[idx]);
    total_ -= weights_[idx];
    weights_[idx] = 0.0;
    alive_[idx] = false;
    ids_[idx].ptr = nullptr;
    free_list_.push_back(idx);
  }

  double totalWeight() const { return total_; }

  // Continuous uniform on [0, total). When the weight is an integer A count,
  // the probability of a component is A_u / N_i.
  ComponentId sample(std::mt19937_64& rng) const {
    if (!(total_ > 0.0)) throw std::runtime_error("sample on empty sampler");
    const double u = std::generate_canonical<double, std::numeric_limits<double>::digits>(rng);
    double target = u * total_;
    if (!(target < total_)) target = std::nextafter(total_, 0.0);

    int idx1 = fenwickFind(target);  // 前缀和第一次超过 target 的槽
    if (idx1 < 1) idx1 = 1;
    if (idx1 > live_count_) idx1 = live_count_;

    for (int t = 0; t < live_count_; ++t) {  // 偶尔踩到已删除槽，往前挪一格再试
      const int i = idx1 - 1;
      if (alive_[i] && weights_[i] > 0.0) return ids_[i];
      idx1 = idx1 % live_count_ + 1;
    }
    throw std::runtime_error("failed to sample live slot");
  }

  double weightAt(SamplerSlotRef ref) const {
    if (ref.index < 0 || ref.index >= live_count_) return 0.0;
    return alive_[ref.index] ? weights_[ref.index] : 0.0;
  }

 private:
  std::vector<double> weights_;
  std::vector<ComponentId> ids_;
  std::vector<uint8_t> alive_;
  std::vector<double> fenwick_;  // 1-indexed, length capacity_+1
  std::vector<int> free_list_;
  int capacity_ = 0;
  int live_count_ = 0;  // high-water number of ever-created slots (incl. dead)
  int fenwick_msb_ = 0; // highest power-of-two <= capacity_
  double total_ = 0.0;

  void reserve(int cap) {
    capacity_ = cap;
    weights_.assign(capacity_, 0.0);
    ids_.assign(capacity_, ComponentId{});
    alive_.assign(capacity_, 0);
    fenwick_.assign(capacity_ + 1, 0.0);
    fenwick_msb_ = 1;
    while ((fenwick_msb_ << 1) <= capacity_) fenwick_msb_ <<= 1;
  }

  void grow() {
    const int new_cap = capacity_ * 2;
    weights_.resize(new_cap, 0.0);
    ids_.resize(new_cap);
    alive_.resize(new_cap, 0);
    capacity_ = new_cap;
    fenwick_msb_ = 1;
    while ((fenwick_msb_ << 1) <= capacity_) fenwick_msb_ <<= 1;
    // rebuild fenwick
    fenwick_.assign(capacity_ + 1, 0.0);
    for (int i = 0; i < live_count_; ++i) {
      if (!alive_[i] || weights_[i] == 0.0) continue;
      fenwickAdd(i + 1, weights_[i]);
    }
  }

  void fenwickAdd(int i, double delta) {
    const int n = capacity_;
    for (; i <= n; i += i & -i) fenwick_[i] += delta;
  }

  // Slot where the target weight mass falls.
  int fenwickFind(double target) const {
    int idx = 0;
    for (int bit = fenwick_msb_; bit > 0; bit >>= 1) {
      const int next = idx + bit;
      if (next <= capacity_ && fenwick_[next] <= target) {
        target -= fenwick_[next];
        idx = next;
      }
    }
    return idx + 1;
  }
};
