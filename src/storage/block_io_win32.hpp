#pragma once
// Windows DirectFile: FILE_FLAG_NO_BUFFERING + optional IOCP batch reads.
// Harvests with GetQueuedCompletionStatusEx; reuses OVERLAPPED storage.
// Falls back to buffered overlapped I/O if unbuffered open is rejected,
// and to synchronous ReadFile if IOCP setup fails.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace e2lsh {
namespace storage {

inline std::wstring utf8ToWide(const std::string& s) {
  if (s.empty()) return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  if (n <= 0) throw std::runtime_error("utf8ToWide failed");
  std::wstring w(static_cast<std::size_t>(n - 1), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
  return w;
}

inline std::string winErr(DWORD e) { return std::to_string(static_cast<unsigned long>(e)); }

class DirectFile {
 public:
  DirectFile() = default;

  void open(const std::string& path, bool create = true, bool prefer_direct = true) {
    close();
    path_ = path;
    const std::wstring wpath = utf8ToWide(path);
    block_ = detectSector(wpath);
    const DWORD disp = create ? OPEN_ALWAYS : OPEN_EXISTING;
    const DWORD access = GENERIC_READ | GENERIC_WRITE;
    const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE;

    direct_ = false;
    handle_ = INVALID_HANDLE_VALUE;
    if (prefer_direct) {
      handle_ = CreateFileW(wpath.c_str(), access, share, nullptr, disp,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
      if (handle_ != INVALID_HANDLE_VALUE) direct_ = true;
    }
    if (handle_ == INVALID_HANDLE_VALUE) {
      handle_ = CreateFileW(wpath.c_str(), access, share, nullptr, disp, FILE_FLAG_OVERLAPPED,
                            nullptr);
      if (handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("CreateFile failed: " + path + " err=" + winErr(GetLastError()));
      }
    }

    iocp_ = CreateIoCompletionPort(handle_, nullptr, 0, 1);
    async_ = (iocp_ != nullptr && iocp_ != INVALID_HANDLE_VALUE);
    if (!async_ && iocp_ == INVALID_HANDLE_VALUE) iocp_ = nullptr;
  }

  ~DirectFile() { close(); }

  DirectFile(const DirectFile&) = delete;
  DirectFile& operator=(const DirectFile&) = delete;
  DirectFile(DirectFile&&) = delete;
  DirectFile& operator=(DirectFile&&) = delete;

  void close() {
    if (handle_ != INVALID_HANDLE_VALUE) {
      CancelIoEx(handle_, nullptr);
      CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
    }
    if (iocp_) {
      CloseHandle(iocp_);
      iocp_ = nullptr;
    }
    async_ = false;
    direct_ = false;
  }

  bool isDirect() const { return direct_; }
  bool hasIoUring() const { return hasAsyncBatch(); }
  bool hasAsyncBatch() const { return async_; }
  std::size_t blockSize() const { return block_; }
  const IoStats& stats() const { return stats_; }
  IoStats& stats() { return stats_; }

  FileOff size() const {
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(handle_, &sz)) throw std::runtime_error("GetFileSizeEx failed");
    return static_cast<FileOff>(sz.QuadPart);
  }

  void truncate(FileOff bytes) {
    LARGE_INTEGER li;
    li.QuadPart = bytes;
    if (!SetFilePointerEx(handle_, li, nullptr, FILE_BEGIN) || !SetEndOfFile(handle_)) {
      throw std::runtime_error("truncate failed err=" + winErr(GetLastError()));
    }
  }

  void writeAligned(FileOff off, const void* buf, std::size_t len) {
    requireAligned(off, len, "writeAligned");
    auto t0 = std::chrono::high_resolution_clock::now();
    issueSync(true, off, const_cast<void*>(buf), len);
    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.n_write += 1;
    stats_.bytes_write += len;
    stats_.write_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  void readAligned(FileOff off, void* buf, std::size_t len) {
    requireAligned(off, len, "readAligned");
    auto t0 = std::chrono::high_resolution_clock::now();
    issueSync(false, off, buf, len);
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
    for (int i = 0; i < n; ++i)
      requireAligned(offs[static_cast<size_t>(i)], lens[static_cast<size_t>(i)], "readManyAligned");
    auto t0 = std::chrono::high_resolution_clock::now();
    bool used_async = false;
    if (async_) used_async = readManyIocp(offs, lens, bufs);
    if (!used_async) {
      order_.resize(static_cast<size_t>(n));
      std::iota(order_.begin(), order_.end(), 0);
      std::sort(order_.begin(), order_.end(),
                [&](int a, int b) { return offs[static_cast<size_t>(a)] < offs[static_cast<size_t>(b)]; });
      for (int k = 0; k < n; ++k) {
        const int i = order_[static_cast<size_t>(k)];
        issueSync(false, offs[static_cast<size_t>(i)], bufs[static_cast<size_t>(i)],
                  lens[static_cast<size_t>(i)]);
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.n_read += static_cast<uint64_t>(n);
    for (int i = 0; i < n; ++i) stats_.bytes_read += lens[static_cast<size_t>(i)];
    stats_.read_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  void fadviseDontNeed() {
    // No POSIX_FADV_DONTNEED analogue required for correctness.
  }

 private:
  static std::size_t detectSector(const std::wstring& wpath) {
    wchar_t root[4] = {L'\0', L':', L'\\', L'\0'};
    const wchar_t* query = nullptr;
    if (wpath.size() >= 2 && wpath[1] == L':') {
      root[0] = wpath[0];
      query = root;
    }
    DWORD spc = 0, bps = 0, nfc = 0, tc = 0;
    if (GetDiskFreeSpaceW(query, &spc, &bps, &nfc, &tc) && bps >= 512) {
      return static_cast<std::size_t>(bps);
    }
    return 4096;
  }

  void requireAligned(FileOff off, std::size_t len, const char* what) const {
    const FileOff b = static_cast<FileOff>(block_);
    if (b <= 0 || off % b != 0 || len % block_ != 0)
      throw std::runtime_error(std::string(what) + " requires block alignment (block=" +
                               std::to_string(block_) + ")");
  }

  // hEvent low bit set: skip IOCP packet for this request (sync path).
  void issueSync(bool is_write, FileOff off, void* buf, std::size_t len) {
    HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ev) throw std::runtime_error("CreateEvent failed");
    char* p = static_cast<char*>(buf);
    std::size_t left = len;
    FileOff o = off;
    while (left) {
      const DWORD chunk = static_cast<DWORD>(std::min(left, static_cast<std::size_t>(1u << 30)));
      OVERLAPPED ov;
      std::memset(&ov, 0, sizeof(ov));
      LARGE_INTEGER li;
      li.QuadPart = o;
      ov.Offset = li.LowPart;
      ov.OffsetHigh = li.HighPart;
      ov.hEvent = reinterpret_cast<HANDLE>(reinterpret_cast<UINT_PTR>(ev) | 1);
      ResetEvent(ev);
      DWORD got = 0;
      BOOL ok = is_write ? WriteFile(handle_, p, chunk, &got, &ov)
                         : ReadFile(handle_, p, chunk, &got, &ov);
      if (!ok) {
        const DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
          CloseHandle(ev);
          throw std::runtime_error(std::string(is_write ? "WriteFile" : "ReadFile") +
                                   " failed err=" + winErr(err));
        }
        if (!GetOverlappedResult(handle_, &ov, &got, TRUE)) {
          const DWORD e2 = GetLastError();
          CloseHandle(ev);
          throw std::runtime_error("GetOverlappedResult failed err=" + winErr(e2));
        }
      }
      if (got == 0) {
        CloseHandle(ev);
        throw std::runtime_error(is_write ? "WriteFile zero" : "ReadFile EOF");
      }
      p += got;
      o += static_cast<FileOff>(got);
      left -= static_cast<std::size_t>(got);
    }
    CloseHandle(ev);
  }

  struct Ov : OVERLAPPED {
    int index;
  };

  static constexpr ULONG kIocpHarvest = 128;
  static constexpr int kMaxInflight = 64;

  bool harvestIocp(int issued, int& completed, bool& had_error) {
    if (iocp_entries_.size() < kIocpHarvest) iocp_entries_.resize(kIocpHarvest);
    while (completed < issued) {
      if (!harvestIocpSome(issued, completed, had_error)) return false;
    }
    return true;
  }

  bool harvestIocpSome(int issued, int& completed, bool& had_error) {
    if (completed >= issued) return true;
    if (iocp_entries_.size() < kIocpHarvest) iocp_entries_.resize(kIocpHarvest);
    const ULONG want = std::min(static_cast<ULONG>(issued - completed),
                                static_cast<ULONG>(iocp_entries_.size()));
    ULONG got = 0;
    if (!GetQueuedCompletionStatusEx(iocp_, iocp_entries_.data(), want, &got, INFINITE, FALSE) ||
        got == 0) {
      return false;
    }
    for (ULONG k = 0; k < got; ++k) {
      LPOVERLAPPED pov = iocp_entries_[k].lpOverlapped;
      if (!pov) return false;
      if (pov->Internal != 0) had_error = true;
      ++completed;
    }
    return true;
  }

  void cancelAndHarvest(int issued, int completed) {
    if (completed >= issued) return;
    CancelIoEx(handle_, nullptr);
    bool ignored = false;
    (void)harvestIocp(issued, completed, ignored);
  }

  bool issueOne(int i, const std::vector<FileOff>& offs, const std::vector<std::size_t>& lens,
                const std::vector<void*>& bufs) {
    if (lens[static_cast<size_t>(i)] > 0xffffffffULL) return false;
    Ov& ov = iocp_ovs_[static_cast<size_t>(i)];
    std::memset(static_cast<OVERLAPPED*>(&ov), 0, sizeof(OVERLAPPED));
    ov.index = i;
    LARGE_INTEGER li;
    li.QuadPart = offs[static_cast<size_t>(i)];
    ov.Offset = li.LowPart;
    ov.OffsetHigh = li.HighPart;
    DWORD got = 0;
    BOOL ok = ReadFile(handle_, bufs[static_cast<size_t>(i)],
                       static_cast<DWORD>(lens[static_cast<size_t>(i)]), &got, &ov);
    return ok || GetLastError() == ERROR_IO_PENDING;
  }

  bool readManyIocp(const std::vector<FileOff>& offs, const std::vector<std::size_t>& lens,
                    const std::vector<void*>& bufs) {
    const int n = static_cast<int>(offs.size());
    if (n <= 0) return true;
    if (iocp_ovs_.size() < static_cast<size_t>(n)) iocp_ovs_.resize(static_cast<size_t>(n));
    if (iocp_entries_.size() < kIocpHarvest) iocp_entries_.resize(kIocpHarvest);
    order_.resize(static_cast<size_t>(n));
    std::iota(order_.begin(), order_.end(), 0);
    std::sort(order_.begin(), order_.end(),
              [&](int a, int b) { return offs[static_cast<size_t>(a)] < offs[static_cast<size_t>(b)]; });

    int next = 0;
    int issued = 0;
    int completed = 0;
    bool had_error = false;
    while (completed < n) {
      while (issued - completed < kMaxInflight && next < n) {
        const int i = order_[static_cast<size_t>(next)];
        if (!issueOne(i, offs, lens, bufs)) {
          cancelAndHarvest(issued, completed);
          return false;
        }
        ++next;
        ++issued;
      }
      if (!harvestIocpSome(issued, completed, had_error)) {
        cancelAndHarvest(issued, completed);
        return false;
      }
      if (had_error) {
        cancelAndHarvest(issued, completed);
        return false;
      }
    }
    return true;
  }

  std::string path_;
  HANDLE handle_ = INVALID_HANDLE_VALUE;
  HANDLE iocp_ = nullptr;
  bool direct_ = false;
  bool async_ = false;
  std::size_t block_ = 4096;
  IoStats stats_;
  std::vector<Ov> iocp_ovs_;
  std::vector<OVERLAPPED_ENTRY> iocp_entries_;
  std::vector<int> order_;
};

}  // namespace storage
}  // namespace e2lsh
