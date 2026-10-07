#pragma once
// Unique-coordinate file: one block-aligned slot per vector (O_DIRECT / NO_BUFFERING).
// Query gathers T slots via async batch / aligned read; the estimator never keeps A_flat_ in heap.

#include "block_io.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace e2lsh {
namespace storage {

class CoordStore {
 public:
  void open(const std::string& path, int dim) {
    if (dim <= 0) throw std::invalid_argument("CoordStore dim");
    dim_ = dim;
    bytes_per_ = static_cast<std::size_t>(dim) * sizeof(double);
    file_.open(path, true);
    slot_ = alignUp(bytes_per_, file_.blockSize());
    const FileOff sz = file_.size();
    if (sz > 0) {
      if (sz % static_cast<FileOff>(slot_) != 0)
        throw std::runtime_error("CoordStore: file size not a multiple of slot");
      n_ = static_cast<uint32_t>(static_cast<std::size_t>(sz) / slot_);
      n_flushed_ = n_;
    } else {
      n_ = 0;
      n_flushed_ = 0;
    }
    slot_buf_ = static_cast<char*>(allocAligned(slot_, file_.blockSize()));
    const std::size_t buf_bytes = std::max(slot_ * 64, alignUp(64u << 20, file_.blockSize()));
    buf_slots_ = buf_bytes / slot_;
    write_buf_[0] = static_cast<char*>(allocAligned(buf_slots_ * slot_, file_.blockSize()));
    write_buf_[1] = static_cast<char*>(allocAligned(buf_slots_ * slot_, file_.blockSize()));
    buf_fill_ = 0;
    active_ = 0;
  }

  ~CoordStore() {
    try {
      flush();
    } catch (...) {
    }
    if (slot_buf_) {
      freeAligned(slot_buf_);
      slot_buf_ = nullptr;
    }
    for (int i = 0; i < 2; ++i) {
      if (write_buf_[i]) {
        freeAligned(write_buf_[i]);
        write_buf_[i] = nullptr;
      }
    }
  }

  CoordStore(const CoordStore&) = delete;
  CoordStore& operator=(const CoordStore&) = delete;
  CoordStore() = default;

  int dim() const { return dim_; }
  uint32_t size() const { return n_; }
  std::size_t slotBytes() const { return slot_; }

  void reserve(uint32_t nslots) {
    waitFlush();
    const FileOff need = static_cast<FileOff>(static_cast<std::size_t>(nslots) * slot_);
    if (file_.size() < need) file_.truncate(need);
  }
  DirectFile& file() { return file_; }
  const DirectFile& file() const { return file_; }

  uint32_t append(const double* x) {
    if (buf_fill_ == buf_slots_) startAsyncFlush();
    char* dst = write_buf_[active_] + buf_fill_ * slot_;
    std::memcpy(dst, x, bytes_per_);
    if (slot_ > bytes_per_) std::memset(dst + bytes_per_, 0, slot_ - bytes_per_);
    ++buf_fill_;
    return n_++;
  }

  void flush() {
    startAsyncFlush();
    waitFlush();
  }

  void readOne(uint32_t idx, double* out) {
    if (idx >= n_) throw std::runtime_error("CoordStore::readOne OOB");
    waitFlush();
    if (idx >= n_flushed_) {
      const uint32_t local = idx - n_flushed_;
      if (local >= buf_fill_) throw std::runtime_error("CoordStore::readOne buf OOB");
      std::memcpy(out, write_buf_[active_] + static_cast<size_t>(local) * slot_, bytes_per_);
      return;
    }
    const FileOff off = static_cast<FileOff>(static_cast<std::size_t>(idx) * slot_);
    file_.readAligned(off, slot_buf_, slot_);
    std::memcpy(out, slot_buf_, bytes_per_);
  }

  void gather(const uint32_t* idxs, int n, double* dst) {
    if (n <= 0) return;
    waitFlush();
    disk_i_.clear();
    disk_i_.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      const uint32_t id = idxs[static_cast<size_t>(i)];
      if (id >= n_) throw std::runtime_error("CoordStore::gather OOB");
      if (id >= n_flushed_) {
        const uint32_t local = id - n_flushed_;
        if (local >= buf_fill_) throw std::runtime_error("CoordStore::gather buf OOB");
        std::memcpy(dst + static_cast<size_t>(i) * static_cast<size_t>(dim_),
                    write_buf_[active_] + static_cast<size_t>(local) * slot_, bytes_per_);
      } else {
        disk_i_.push_back(i);
      }
    }
    if (disk_i_.empty()) return;

    std::sort(disk_i_.begin(), disk_i_.end(),
              [&](int a, int b) { return idxs[static_cast<size_t>(a)] < idxs[static_cast<size_t>(b)]; });

    uniq_ids_.clear();
    uniq_ids_.reserve(disk_i_.size());
    hit_u_.resize(disk_i_.size());
    for (size_t j = 0; j < disk_i_.size(); ++j) {
      const uint32_t id = idxs[static_cast<size_t>(disk_i_[j])];
      if (uniq_ids_.empty() || uniq_ids_.back() != id) uniq_ids_.push_back(id);
      hit_u_[j] = static_cast<int>(uniq_ids_.size()) - 1;
    }

    const int u = static_cast<int>(uniq_ids_.size());
    const std::size_t bsz = file_.blockSize();
    char* base = static_cast<char*>(gather_scratch_.ensure(static_cast<size_t>(u) * slot_, bsz));
    gather_offs_.clear();
    gather_lens_.clear();
    gather_bufs_.clear();
    int r = 0;
    while (r < u) {
      int s = r;
      while (r + 1 < u && uniq_ids_[static_cast<size_t>(r) + 1] == uniq_ids_[static_cast<size_t>(r)] + 1)
        ++r;
      const int run = r - s + 1;
      gather_offs_.push_back(static_cast<FileOff>(static_cast<size_t>(uniq_ids_[static_cast<size_t>(s)]) * slot_));
      gather_lens_.push_back(static_cast<size_t>(run) * slot_);
      gather_bufs_.push_back(base + static_cast<size_t>(s) * slot_);
      ++r;
    }
    file_.readManyAligned(gather_offs_, gather_lens_, gather_bufs_);
    for (size_t j = 0; j < disk_i_.size(); ++j) {
      const int i = disk_i_[j];
      std::memcpy(dst + static_cast<size_t>(i) * static_cast<size_t>(dim_),
                  base + static_cast<size_t>(hit_u_[j]) * slot_, bytes_per_);
    }
  }

  void fadviseDontNeed() {
    flush();
    file_.fadviseDontNeed();
  }

 private:
  void waitFlush() {
    if (!flush_thr_.joinable()) return;
    flush_thr_.join();
    n_flushed_ += inflight_slots_;
    inflight_slots_ = 0;
    inflight_buf_ = nullptr;
    if (flush_ex_) {
      std::exception_ptr e = flush_ex_;
      flush_ex_ = nullptr;
      std::rethrow_exception(e);
    }
  }

  void startAsyncFlush() {
    waitFlush();
    if (buf_fill_ == 0) return;
    const FileOff off = static_cast<FileOff>(static_cast<std::size_t>(n_flushed_) * slot_);
    const std::size_t bytes = buf_fill_ * slot_;
    const FileOff need = off + static_cast<FileOff>(bytes);
    if (file_.size() < need) file_.truncate(need);
    inflight_buf_ = write_buf_[active_];
    inflight_off_ = off;
    inflight_bytes_ = bytes;
    inflight_slots_ = static_cast<uint32_t>(buf_fill_);
    active_ ^= 1;
    buf_fill_ = 0;
    flush_ex_ = nullptr;
    flush_thr_ = std::thread([this]() {
      try {
        file_.writeAligned(inflight_off_, inflight_buf_, inflight_bytes_);
      } catch (...) {
        flush_ex_ = std::current_exception();
      }
    });
  }

  DirectFile file_;
  int dim_ = 0;
  std::size_t bytes_per_ = 0;
  std::size_t slot_ = 0;
  uint32_t n_ = 0;
  uint32_t n_flushed_ = 0;
  char* slot_buf_ = nullptr;
  char* write_buf_[2] = {nullptr, nullptr};
  int active_ = 0;
  std::size_t buf_slots_ = 0;
  std::size_t buf_fill_ = 0;

  std::thread flush_thr_;
  std::exception_ptr flush_ex_;
  char* inflight_buf_ = nullptr;
  FileOff inflight_off_ = 0;
  std::size_t inflight_bytes_ = 0;
  uint32_t inflight_slots_ = 0;

  e2lsh::os::AlignedScratch gather_scratch_;
  std::vector<int> disk_i_;
  std::vector<uint32_t> uniq_ids_;
  std::vector<int> hit_u_;
  std::vector<FileOff> gather_offs_;
  std::vector<std::size_t> gather_lens_;
  std::vector<void*> gather_bufs_;
};

}  // namespace storage
}  // namespace e2lsh
