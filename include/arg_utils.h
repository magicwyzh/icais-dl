#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>
namespace icdl{

#define OP_ADD_TENSOR(name) \
    public:\
        const Tensor& get_##name() const{\
            return _##name;\
        }\
    private:\
        Tensor _##name

#define OP_ADD_OPTIONS(op_name) \
    public:\
        const op_name##Options& get_options() const{\
            return _options;\
        }\
    private:\
        op_name##Options _options

#define OP_ADD_CONSTRUCTOR_WITH_OPTION_DECLARATION(op_name)

#define ICDL_ARG(T, name)                                       \
  auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name;                                  \
    return *this;                                                \
  }                                                              \
  auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                       \
    return *this;                                                \
  }                                                              \
  const T& name() const noexcept { /* NOLINT */                  \
    return this->name##_;                                        \
  }                                                              \
  T name##_ /* NOLINT */




/// A utility class that accepts either a container of `D`-many values, or a
/// single value, which is internally repeated `D` times. This is useful to
/// represent parameters that are multidimensional, but often equally sized in
/// all dimensions. For example, the kernel size of a 2D convolution has an `x`
/// and `y` length, but `x` and `y` are often equal. In such a case you could
/// just pass `3` to an `ExpandingArray<2>` and it would "expand" to `{3, 3}`.
template <size_t D, typename T = int64_t>
class ExpandingArray {
 public:
  /// Constructs an `ExpandingArray` from an `initializer_list`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(std::initializer_list<T> list)
      : ExpandingArray(std::vector<T>(list)) {}

  /// Constructs an `ExpandingArray` from an `initializer_list`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(std::vector<T> values) {
    assert(values.size() == D);
    std::copy(values.begin(), values.end(), values_.begin());
  }

  /// Constructs an `ExpandingArray` from a single value, which is repeated `D`
  /// times (where `D` is the extent parameter of the `ExpandingArray`).
  /*implicit*/ ExpandingArray(T single_size) {
    values_.fill(single_size);
  }

  /// Constructs an `ExpandingArray` from a correctly sized `std::array`.
  /*implicit*/ ExpandingArray(const std::array<T, D>& values)
      : values_(values) {}

  /// Accesses the underlying `std::array`.
  std::array<T, D>& operator*() {
    return values_;
  }

  /// Accesses the underlying `std::array`.
  const std::array<T, D>& operator*() const {
    return values_;
  }

  /// Accesses the underlying `std::array`.
  std::array<T, D>* operator->() {
    return &values_;
  }

  /// Accesses the underlying `std::array`.
  const std::array<T, D>* operator->() const {
    return &values_;
  }

  /// Returns an `ArrayRef` to the underlying `std::array`.

  operator std::array<T, D>() const{
    return values_;
  }

  /// Returns the extent of the `ExpandingArray`.
  size_t size() const noexcept {
    return D;
  }

 private:
  /// The backing array.
  std::array<T, D> values_;
};

template <size_t D, typename T>
std::ostream& operator<<(
    std::ostream& stream,
    const ExpandingArray<D, T>& expanding_array) {
  if (expanding_array.size() == 1) {
    return stream << expanding_array->at(0);
  }
  return stream << static_cast<std::array<T,D>>(expanding_array);
}


}//namespace icdl

