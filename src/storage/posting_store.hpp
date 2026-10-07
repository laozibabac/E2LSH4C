#pragma once
// Packed PointId postings. Build uses a RAM arena (doubling per bucket);
// finalizeA() writes a dense block-aligned file and drops the arena.
// Sample i of bucket b is one aligned block read (O(1) I/O).

#include "block_io.hpp"
#include "../e2lsh_hash.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace e2lsh {
namespace storage {

class PostingStore {
 public:
  void open(const std::string& path) {
    path_ = path;
    file_.open(path, true);
  }

  DirectFile& file() { return file_; }
  const DirectFile& file() const { return file_; }
  bool finalized() const { return finalized_; }

  void ensureBucket(uint32_t b) {
    if (b >= p_len_.size()) {
      const uint32_t n = b + 1;
      p_off_.resize(n, 0);
      p_len_.resize(n, 0);
      p_cap_.resize(n, 0);
      disk_off_.resize(n, 0);
      disk_len_.resize(n, 0);
    }
  }

  void append(uint32_t b, PointId id) {
    ensureBucket(b);
    if (finalized_) {
      overflow_[b].push_back(id);
      p_len_[b] += 1;
      return;
    }
    if (p_len_[b] == p_cap_[b]) {
      const uint32_t ncap = p_cap_[b] == 0 ? 1u : p_cap_[b] * 2u;
      const uint32_t noff = static_cast<uint32_t>(arena_.size());
      arena_.resize(static_cast<size_t>(noff) + static_cast<size_t>(ncap), 0);
      if (p_len_[b] > 0) {
        std::memcpy(arena_.data() + noff, arena_.data() + p_off_[b],
                    static_cast<size_t>(p_len_[b]) * sizeof(PointId));
      }
      p_off_[b] = noff;
      p_cap_[b] = ncap;
    }
    arena_[p_off_[b] + p_len_[b]] = id;
    p_len_[b] += 1;
  }

  uint32_t length(uint32_t b) const {
    if (b >= p_len_.size()) return 0;
    return p_len_[b];
  }

  PointId at(uint32_t b, uint32_t i) {
    if (b >= p_len_.size() || i >= p_len_[b]) throw std::runtime_error("PostingStore::at OOB");
    const uint32_t packed = disk_len_[b];
    if (finalized_ && i < packed) return readDisk(b, i);
    if (finalized_) {
      auto it = overflow_.find(b);
      if (it == overflow_.end()) throw std::runtime_error("PostingStore overflow missing");
      return it->second[static_cast<size_t>(i - packed)];
    }
    return arena_[p_off_[b] + i];
  }

  void readAll(uint32_t b, std::vector<PointId>& out) {
    out.clear();
    if (b >= p_len_.size() || p_len_[b] == 0) return;
    out.resize(p_len_[b]);
    if (!finalized_) {
      std::memcpy(out.data(), arena_.data() + p_off_[b], static_cast<size_t>(p_len_[b]) * sizeof(PointId));
      return;
    }
    const uint32_t packed = disk_len_[b];
    if (packed > 0) {
      const std::size_t bsz = file_.blockSize();
      const std::size_t bytes = static_cast<std::size_t>(packed) * sizeof(PointId);
      const FileOff byte_off =
          static_cast<FileOff>(disk_off_[b]) * static_cast<FileOff>(sizeof(PointId));
      const FileOff aligned = alignDownOff(byte_off, bsz);
      const std::size_t span = alignUp(static_cast<std::size_t>(byte_off - aligned) + bytes, bsz);
      char* buf = static_cast<char*>(allocAligned(span, bsz));
      file_.readAligned(aligned, buf, span);
      std::memcpy(out.data(), buf + (byte_off - aligned), bytes);
      freeAligned(buf);
    }
    if (p_len_[b] > packed) {
      auto it = overflow_.find(b);
      if (it == overflow_.end()) throw std::runtime_error("PostingStore overflow missing");
      std::memcpy(out.data() + packed, it->second.data(),
                  static_cast<size_t>(p_len_[b] - packed) * sizeof(PointId));
    }
  }

  void gather(const uint32_t* buckets, const uint32_t* slots, int n, PointId* out) {
    if (n <= 0) return;
    if (!finalized_) {
      for (int i = 0; i < n; ++i) out[i] = at(buckets[i], slots[i]);
      return;
    }
    struct Hit {
      FileOff aligned;
      std::size_t within;
      int dst;
    };
    std::vector<Hit> hits;
    hits.reserve(static_cast<size_t>(n));
    const std::size_t bsz = file_.blockSize();
    for (int i = 0; i < n; ++i) {
      const uint32_t b = buckets[i];
      const uint32_t s = slots[i];
      if (b >= p_len_.size() || s >= p_len_[b]) throw std::runtime_error("PostingStore::gather OOB");
      if (s < disk_len_[b]) {
        const FileOff byte_off =
            static_cast<FileOff>(disk_off_[b] + s) * static_cast<FileOff>(sizeof(PointId));
        const FileOff aligned = alignDownOff(byte_off, bsz);
        hits.push_back(Hit{aligned, static_cast<std::size_t>(byte_off - aligned), i});
      } else {
        out[i] = at(b, s);
      }
    }
    if (hits.empty()) return;
    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
      if (a.aligned != b.aligned) return a.aligned < b.aligned;
      return a.dst < b.dst;
    });
    std::vector<FileOff> uniq;
    uniq.reserve(hits.size());
    std::vector<int> hit_u(hits.size());
    for (size_t j = 0; j < hits.size(); ++j) {
      if (uniq.empty() || uniq.back() != hits[j].aligned) uniq.push_back(hits[j].aligned);
      hit_u[j] = static_cast<int>(uniq.size()) - 1;
    }
    const int u = static_cast<int>(uniq.size());
    char* base = static_cast<char*>(gather_scratch_.ensure(static_cast<size_t>(u) * bsz, bsz));
    gather_offs_.clear();
    gather_lens_.clear();
    gather_bufs_.clear();
    int r = 0;
    while (r < u) {
      int s = r;
      while (r + 1 < u &&
             uniq[static_cast<size_t>(r) + 1] == uniq[static_cast<size_t>(r)] + static_cast<FileOff>(bsz))
        ++r;
      const int run = r - s + 1;
      gather_offs_.push_back(uniq[static_cast<size_t>(s)]);
      gather_lens_.push_back(static_cast<size_t>(run) * bsz);
      gather_bufs_.push_back(base + static_cast<size_t>(s) * bsz);
      ++r;
    }
    file_.readManyAligned(gather_offs_, gather_lens_, gather_bufs_);
    for (size_t j = 0; j < hits.size(); ++j) {
      PointId id;
      std::memcpy(&id, base + static_cast<size_t>(hit_u[j]) * bsz + hits[j].within, sizeof(PointId));
      out[hits[j].dst] = id;
    }
  }

  void finalize() {
    if (finalized_ && arena_.empty()) return;
    const uint32_t nb = static_cast<uint32_t>(p_len_.size());
    uint64_t total = 0;
    for (uint32_t b = 0; b < nb; ++b) total += p_len_[b];
    const std::size_t bsz = file_.blockSize();
    const std::size_t bytes = static_cast<std::size_t>(total) * sizeof(PointId);
    const std::size_t padded = alignUp(std::max(bytes, bsz), bsz);
    file_.truncate(static_cast<FileOff>(padded));
    char* buf = static_cast<char*>(allocAligned(padded, bsz));
    uint64_t cursor = 0;
    for (uint32_t b = 0; b < nb; ++b) {
      disk_off_[b] = static_cast<uint32_t>(cursor);
      disk_len_[b] = p_len_[b];
      if (p_len_[b] > 0) {
        std::memcpy(buf + cursor * sizeof(PointId), arena_.data() + p_off_[b],
                    static_cast<size_t>(p_len_[b]) * sizeof(PointId));
        cursor += p_len_[b];
      }
    }
    file_.writeAligned(0, buf, padded);
    freeAligned(buf);
    arena_.clear();
    arena_.shrink_to_fit();
    p_off_.clear();
    p_off_.shrink_to_fit();
    p_cap_.clear();
    p_cap_.shrink_to_fit();
    finalized_ = true;
    file_.fadviseDontNeed();
  }

  void dropArena() {
    arena_.clear();
    arena_.shrink_to_fit();
  }

 private:
  PointId readDisk(uint32_t b, uint32_t i) {
    PointId id = 0;
    gather(&b, &i, 1, &id);
    return id;
  }

  DirectFile file_;
  std::string path_;
  bool finalized_ = false;
  std::vector<PointId> arena_;
  std::vector<uint32_t> p_off_, p_len_, p_cap_;
  std::vector<uint32_t> disk_off_, disk_len_;
  std::unordered_map<uint32_t, std::vector<PointId>> overflow_;
  e2lsh::os::AlignedScratch gather_scratch_;
  std::vector<FileOff> gather_offs_;
  std::vector<std::size_t> gather_lens_;
  std::vector<void*> gather_bufs_;
};

}  // namespace storage
}  // namespace e2lsh
