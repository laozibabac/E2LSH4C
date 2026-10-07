#pragma once
// Direct, sector-aligned file I/O.
// POSIX: O_DIRECT + optional io_uring (pread/pwrite fallback).
// Windows: FILE_FLAG_NO_BUFFERING + optional IOCP / GetQueuedCompletionStatusEx
// (synchronous overlapped fallback).

#include "../os/aligned_alloc.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace e2lsh {
namespace storage {

using FileOff = std::int64_t;

// Default Linux O_DIRECT unit. Windows uses the volume sector size at open().
constexpr std::size_t kBlock = 512;

inline std::size_t alignUp(std::size_t n, std::size_t a = kBlock) {
  return e2lsh::os::alignUp(n, a);
}

inline void* allocAligned(std::size_t bytes, std::size_t align = kBlock) {
  return e2lsh::os::allocAligned(bytes, align);
}

inline void freeAligned(void* p) { e2lsh::os::freeAligned(p); }

inline FileOff alignDownOff(FileOff off, std::size_t a) {
  const FileOff mask = static_cast<FileOff>(a) - 1;
  return off & ~mask;
}

struct IoStats {
  uint64_t n_read = 0;
  uint64_t n_write = 0;
  uint64_t bytes_read = 0;
  uint64_t bytes_write = 0;
  double read_ms = 0.0;
  double write_ms = 0.0;
};

}  // namespace storage
}  // namespace e2lsh

#ifdef _WIN32
#include "block_io_win32.hpp"
#else
#include "block_io_posix.hpp"
#endif
