#pragma once
// Read-only file mapping: mmap (POSIX) / MapViewOfFile (Windows).

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace e2lsh {
namespace os {

class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile() { close(); }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& o) noexcept { *this = std::move(o); }
  MappedFile& operator=(MappedFile&& o) noexcept {
    if (this == &o) return *this;
    close();
#ifdef _WIN32
    view_ = o.view_;
    map_ = o.map_;
    file_ = o.file_;
    size_ = o.size_;
    o.view_ = nullptr;
    o.map_ = nullptr;
    o.file_ = INVALID_HANDLE_VALUE;
    o.size_ = 0;
#else
    ptr_ = o.ptr_;
    size_ = o.size_;
    fd_ = o.fd_;
    o.ptr_ = nullptr;
    o.size_ = 0;
    o.fd_ = -1;
#endif
    return *this;
  }

  bool openRead(const std::string& path) {
    close();
#ifdef _WIN32
    int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (n <= 0) return false;
    std::wstring w(static_cast<std::size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
    file_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file_, &sz) || sz.QuadPart <= 0) {
      close();
      return false;
    }
    size_ = static_cast<std::size_t>(sz.QuadPart);
    map_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map_) {
      close();
      return false;
    }
    view_ = MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
    if (!view_) {
      close();
      return false;
    }
    return true;
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;
    struct stat st {};
    if (fstat(fd_, &st) != 0 || st.st_size <= 0) {
      close();
      return false;
    }
    size_ = static_cast<std::size_t>(st.st_size);
    ptr_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr_ == MAP_FAILED) {
      ptr_ = nullptr;
      close();
      return false;
    }
#ifdef POSIX_MADV_SEQUENTIAL
    posix_madvise(ptr_, size_, POSIX_MADV_SEQUENTIAL);
#elif defined(MADV_SEQUENTIAL)
    madvise(ptr_, size_, MADV_SEQUENTIAL);
#endif
    return true;
#endif
  }

  void close() {
#ifdef _WIN32
    if (view_) {
      UnmapViewOfFile(view_);
      view_ = nullptr;
    }
    if (map_) {
      CloseHandle(map_);
      map_ = nullptr;
    }
    if (file_ != INVALID_HANDLE_VALUE) {
      CloseHandle(file_);
      file_ = INVALID_HANDLE_VALUE;
    }
    size_ = 0;
#else
    if (ptr_ && ptr_ != MAP_FAILED) {
      munmap(ptr_, size_);
      ptr_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    size_ = 0;
#endif
  }

  bool valid() const {
#ifdef _WIN32
    return view_ != nullptr && size_ > 0;
#else
    return ptr_ != nullptr && ptr_ != MAP_FAILED && size_ > 0;
#endif
  }

  const void* data() const {
#ifdef _WIN32
    return view_;
#else
    return ptr_;
#endif
  }

  std::size_t size() const { return size_; }

 private:
#ifdef _WIN32
  HANDLE file_ = INVALID_HANDLE_VALUE;
  HANDLE map_ = nullptr;
  void* view_ = nullptr;
  std::size_t size_ = 0;
#else
  int fd_ = -1;
  void* ptr_ = nullptr;
  std::size_t size_ = 0;
#endif
};

}  // namespace os
}  // namespace e2lsh
