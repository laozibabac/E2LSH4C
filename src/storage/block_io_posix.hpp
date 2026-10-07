#pragma once
// POSIX DirectFile: O_DIRECT + optional io_uring batch (raw syscalls, no liburing).
// Falls back to aligned pread/pwrite if io_uring setup fails or O_DIRECT is rejected.

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace e2lsh {
namespace storage {

class IoUring {
 public:
  explicit IoUring(unsigned entries = 256) {
    struct io_uring_params p;
    std::memset(&p, 0, sizeof(p));
    ring_fd_ = static_cast<int>(syscall(__NR_io_uring_setup, entries, &p));
    if (ring_fd_ < 0) return;

    std::size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    std::size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
      if (cq_sz > sq_sz) sq_sz = cq_sz;
      cq_sz = sq_sz;
    }
    sq_ring_ = mmap(nullptr, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd_,
                    IORING_OFF_SQ_RING);
    if (sq_ring_ == MAP_FAILED) {
      close(ring_fd_);
      ring_fd_ = -1;
      sq_ring_ = nullptr;
      return;
    }
    sq_map_sz_ = sq_sz;
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
      cq_ring_ = sq_ring_;
      cq_map_sz_ = 0;
    } else {
      cq_ring_ = mmap(nullptr, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd_,
                      IORING_OFF_CQ_RING);
      if (cq_ring_ == MAP_FAILED) {
        munmap(sq_ring_, sq_map_sz_);
        close(ring_fd_);
        ring_fd_ = -1;
        sq_ring_ = nullptr;
        cq_ring_ = nullptr;
        return;
      }
      cq_map_sz_ = cq_sz;
    }
    sqes_ = mmap(nullptr, p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_SQES);
    if (sqes_ == MAP_FAILED) {
      if (cq_map_sz_) munmap(cq_ring_, cq_map_sz_);
      munmap(sq_ring_, sq_map_sz_);
      close(ring_fd_);
      ring_fd_ = -1;
      sqes_ = nullptr;
      return;
    }
    sqe_map_sz_ = p.sq_entries * sizeof(struct io_uring_sqe);
    sq_head_ = reinterpret_cast<unsigned*>(static_cast<char*>(sq_ring_) + p.sq_off.head);
    sq_tail_ = reinterpret_cast<unsigned*>(static_cast<char*>(sq_ring_) + p.sq_off.tail);
    sq_mask_ = reinterpret_cast<unsigned*>(static_cast<char*>(sq_ring_) + p.sq_off.ring_mask);
    sq_array_ = reinterpret_cast<unsigned*>(static_cast<char*>(sq_ring_) + p.sq_off.array);
    cq_head_ = reinterpret_cast<unsigned*>(static_cast<char*>(cq_ring_) + p.cq_off.head);
    cq_tail_ = reinterpret_cast<unsigned*>(static_cast<char*>(cq_ring_) + p.cq_off.tail);
    cq_mask_ = reinterpret_cast<unsigned*>(static_cast<char*>(cq_ring_) + p.cq_off.ring_mask);
    cqes_ = reinterpret_cast<struct io_uring_cqe*>(static_cast<char*>(cq_ring_) + p.cq_off.cqes);
    entries_ = p.sq_entries;
    ok_ = true;
  }

  ~IoUring() { closeRing(); }

  IoUring(const IoUring&) = delete;
  IoUring& operator=(const IoUring&) = delete;

  bool ok() const { return ok_; }
  unsigned entries() const { return entries_; }

  bool readMany(int fd, const FileOff* offs, const std::size_t* lens, void** bufs, int n) {
    if (!ok_ || n <= 0) return false;
    int done = 0;
    while (done < n) {
      const int batch = std::min(n - done, static_cast<int>(entries_));
      unsigned tail = *sq_tail_;
      for (int i = 0; i < batch; ++i) {
        const unsigned idx = tail & *sq_mask_;
        struct io_uring_sqe* sqe = &static_cast<struct io_uring_sqe*>(sqes_)[idx];
        std::memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ;
        sqe->fd = fd;
        sqe->off = static_cast<uint64_t>(offs[done + i]);
        sqe->addr = reinterpret_cast<uint64_t>(bufs[done + i]);
        sqe->len = static_cast<uint32_t>(lens[done + i]);
        sqe->user_data = static_cast<uint64_t>(done + i);
        sq_array_[idx] = idx;
        ++tail;
      }
      *sq_tail_ = tail;
      int ret = static_cast<int>(
          syscall(__NR_io_uring_enter, ring_fd_, batch, batch, IORING_ENTER_GETEVENTS, nullptr, 0));
      if (ret < 0) return false;
      unsigned head = *cq_head_;
      for (int i = 0; i < batch; ++i) {
        struct io_uring_cqe* cqe = &cqes_[head & *cq_mask_];
        if (cqe->res < 0) return false;
        ++head;
      }
      *cq_head_ = head;
      done += batch;
    }
    return true;
  }

 private:
  void closeRing() {
    if (sqes_ && sqes_ != MAP_FAILED) munmap(sqes_, sqe_map_sz_);
    if (cq_map_sz_ && cq_ring_ && cq_ring_ != MAP_FAILED) munmap(cq_ring_, cq_map_sz_);
    if (sq_ring_ && sq_ring_ != MAP_FAILED) munmap(sq_ring_, sq_map_sz_);
    if (ring_fd_ >= 0) close(ring_fd_);
    ok_ = false;
    ring_fd_ = -1;
    sq_ring_ = cq_ring_ = sqes_ = nullptr;
  }

  bool ok_ = false;
  int ring_fd_ = -1;
  unsigned entries_ = 0;
  void* sq_ring_ = nullptr;
  void* cq_ring_ = nullptr;
  void* sqes_ = nullptr;
  std::size_t sq_map_sz_ = 0, cq_map_sz_ = 0, sqe_map_sz_ = 0;
  unsigned* sq_head_ = nullptr;
  unsigned* sq_tail_ = nullptr;
  unsigned* sq_mask_ = nullptr;
  unsigned* sq_array_ = nullptr;
  unsigned* cq_head_ = nullptr;
  unsigned* cq_tail_ = nullptr;
  unsigned* cq_mask_ = nullptr;
  struct io_uring_cqe* cqes_ = nullptr;
};

class DirectFile {
 public:
  DirectFile() = default;

  void open(const std::string& path, bool create = true, bool prefer_direct = true) {
    close();
    path_ = path;
    block_ = kBlock;
    int flags = O_RDWR;
    if (create) flags |= O_CREAT;
    fd_ = -1;
    fd_direct_ = -1;
    direct_ = false;
    if (prefer_direct) {
      fd_direct_ = ::open(path.c_str(), flags | O_DIRECT, 0644);
      if (fd_direct_ >= 0) {
        direct_ = true;
        fd_ = fd_direct_;
      }
    }
    if (fd_ < 0) {
      fd_ = ::open(path.c_str(), flags, 0644);
      if (fd_ < 0)
        throw std::runtime_error("open failed: " + path + " errno=" + std::to_string(errno));
      direct_ = false;
      fd_direct_ = -1;
    }
    ring_ = std::make_unique<IoUring>(128);
  }

  ~DirectFile() { close(); }

  DirectFile(const DirectFile&) = delete;
  DirectFile& operator=(const DirectFile&) = delete;
  DirectFile(DirectFile&&) = delete;
  DirectFile& operator=(DirectFile&&) = delete;

  void close() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
      fd_direct_ = -1;
    }
    ring_.reset();
  }

  bool isDirect() const { return direct_; }
  bool hasIoUring() const { return hasAsyncBatch(); }
  bool hasAsyncBatch() const { return ring_ && ring_->ok(); }
  int fd() const { return fd_; }
  std::size_t blockSize() const { return block_; }
  const IoStats& stats() const { return stats_; }
  IoStats& stats() { return stats_; }

  FileOff size() const {
    struct stat st;
    if (fstat(fd_, &st) != 0) throw std::runtime_error("fstat failed");
    return static_cast<FileOff>(st.st_size);
  }

  void truncate(FileOff bytes) {
    if (ftruncate(fd_, static_cast<off_t>(bytes)) != 0) throw std::runtime_error("ftruncate failed");
  }

  void writeAligned(FileOff off, const void* buf, std::size_t len) {
    requireAligned(off, len, "writeAligned");
    auto t0 = std::chrono::high_resolution_clock::now();
    const char* p = static_cast<const char*>(buf);
    std::size_t left = len;
    FileOff o = off;
    while (left) {
      ssize_t n = pwrite(fd_, p, left, static_cast<off_t>(o));
      if (n < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("pwrite failed errno=" + std::to_string(errno));
      }
      p += n;
      o += n;
      left -= static_cast<std::size_t>(n);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.n_write += 1;
    stats_.bytes_write += len;
    stats_.write_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  void readAligned(FileOff off, void* buf, std::size_t len) {
    requireAligned(off, len, "readAligned");
    auto t0 = std::chrono::high_resolution_clock::now();
    preadLoop(off, buf, len);
    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.n_read += 1;
    stats_.bytes_read += len;
    stats_.read_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  void readManyAligned(const std::vector<FileOff>& offs, const std::vector<std::size_t>& lens,
                       const std::vector<void*>& bufs) {
    const int n = static_cast<int>(offs.size());
    if (n == 0) return;
    if (static_cast<int>(lens.size()) != n || static_cast<int>(bufs.size()) != n)
      throw std::runtime_error("readManyAligned size mismatch");
    auto t0 = std::chrono::high_resolution_clock::now();
    bool used_uring = false;
    if (ring_ && ring_->ok()) {
      std::vector<int> order(static_cast<size_t>(n));
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int a, int b) { return offs[static_cast<size_t>(a)] < offs[static_cast<size_t>(b)]; });
      std::vector<FileOff> so(static_cast<size_t>(n));
      std::vector<std::size_t> sl(static_cast<size_t>(n));
      std::vector<void*> sb(static_cast<size_t>(n));
      for (int k = 0; k < n; ++k) {
        const int i = order[static_cast<size_t>(k)];
        so[static_cast<size_t>(k)] = offs[static_cast<size_t>(i)];
        sl[static_cast<size_t>(k)] = lens[static_cast<size_t>(i)];
        sb[static_cast<size_t>(k)] = bufs[static_cast<size_t>(i)];
      }
      used_uring = ring_->readMany(fd_, so.data(), sl.data(), sb.data(), n);
    }
    if (!used_uring) {
      std::vector<int> order(static_cast<size_t>(n));
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int a, int b) { return offs[static_cast<size_t>(a)] < offs[static_cast<size_t>(b)]; });
      for (int k = 0; k < n; ++k) {
        const int i = order[static_cast<size_t>(k)];
        preadLoop(offs[static_cast<size_t>(i)], bufs[static_cast<size_t>(i)],
                  lens[static_cast<size_t>(i)]);
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.n_read += static_cast<uint64_t>(n);
    for (int i = 0; i < n; ++i) stats_.bytes_read += lens[static_cast<size_t>(i)];
    stats_.read_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  void fadviseDontNeed() {
    if (fd_ < 0) return;
#ifdef POSIX_FADV_DONTNEED
    (void)posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
#endif
  }

 private:
  void requireAligned(FileOff off, std::size_t len, const char* what) const {
    const FileOff b = static_cast<FileOff>(block_);
    if (off % b != 0 || len % block_ != 0)
      throw std::runtime_error(std::string(what) + " requires block alignment");
  }

  void preadLoop(FileOff off, void* buf, std::size_t len) {
    char* p = static_cast<char*>(buf);
    std::size_t left = len;
    FileOff o = off;
    while (left) {
      ssize_t n = pread(fd_, p, left, static_cast<off_t>(o));
      if (n < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("pread failed errno=" + std::to_string(errno));
      }
      if (n == 0) throw std::runtime_error("pread EOF");
      p += n;
      o += n;
      left -= static_cast<std::size_t>(n);
    }
  }

  std::string path_;
  int fd_ = -1;
  int fd_direct_ = -1;
  bool direct_ = false;
  std::size_t block_ = kBlock;
  std::unique_ptr<IoUring> ring_;
  IoStats stats_;
};

}  // namespace storage
}  // namespace e2lsh
