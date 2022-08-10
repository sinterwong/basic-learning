#ifndef __UTILS_EXCEPTION_HPP_
#define __UTILS_EXCEPTION_HPP_

#include <stdexcept>
#include <variant>

namespace utils::exception {
template <typename T, typename... Ts>
const T &get_or_throw(const std::variant<Ts...> &v) {
  if (const auto ptr = std::get_if<T>(&v)) {
    return *ptr;
  }
  throw std::runtime_error("Requested type not found in variant");
}

class InvalidValueException : public std::runtime_error {
public:
  explicit InvalidValueException(const std::string &message)
      : std::runtime_error("Invalid value: " + message) {}
};

class OutOfRangeException : public std::out_of_range {
public:
  explicit OutOfRangeException(const std::string &message)
      : std::out_of_range("Out of range: " + message) {}
};

class NullPointerException : public std::logic_error {
public:
  explicit NullPointerException(const std::string &message)
      : std::logic_error("Null pointer: " + message) {}
};

class FileOperationException : public std::runtime_error {
public:
  explicit FileOperationException(const std::string &message)
      : std::runtime_error("File operation error: " + message) {}
};

class NetworkException : public std::runtime_error {
public:
  explicit NetworkException(const std::string &message)
      : std::runtime_error("Network error: " + message) {}
};

} // namespace utils::exception

#endif