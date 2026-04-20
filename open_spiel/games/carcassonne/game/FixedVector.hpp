#include <array>
#include <cassert>
#include <cstdint>

template <typename T, std::size_t N>
class FixedVector {
    std::array<T, N> data_ = {};
    int size_ = 0;

  public:
    int size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T &operator[](int index) { return data_[index]; }
    const T &operator[](int index) const { return data_[index]; }
    void push_back(const T &value) {
        assert(size_ < static_cast<int>(N));
        data_[size_++] = value;
    }
    void swap_pop_erase_at(int index) {
        data_[index] = data_[size_ - 1];
        size_--;
    }
    T *begin() { return data_.data(); }
    T *end() { return data_.data() + size_; }
    const T *begin() const { return data_.data(); }
    const T *end() const { return data_.data() + size_; }
};