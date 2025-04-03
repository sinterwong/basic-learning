/**
 * @file type_safe_factory.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __TYPE_SAFE_FACTORY_HPP__
#define __TYPE_SAFE_FACTORY_HPP__

#include <any>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace oop::factory {

using ConstructorParams = std::map<std::string, std::any>;

template <typename T>
T get_param(const ConstructorParams &params, const std::string &key) {
  auto it = params.find(key);
  if (it == params.end()) {
    throw std::runtime_error("Missing required parameter: " + key);
  }
  try {
    return std::any_cast<T>(it->second);
  } catch (const std::bad_any_cast &e) {
    throw std::runtime_error("Invalid parameter type for key '" + key +
                             "'. Expected type: " + typeid(T).name());
  }
}

template <typename T>
std::optional<T> get_optional_param(const ConstructorParams &params,
                                    const std::string &key) {
  auto it = params.find(key);
  if (it == params.end()) {
    return std::nullopt;
  }

  try {
    return std::any_cast<T>(it->second);
  } catch (const std::bad_any_cast &e) {
    throw std::runtime_error("Invalid parameter type for optional key '" + key +
                             "'. Expected type: " + typeid(T).name());
  }
}

template <class BaseClass> class Factory {
public:
  // takes a const refer to paramters and returns a shared_ptr to the BaseClass
  using Creator =
      std::function<std::shared_ptr<BaseClass>(const ConstructorParams &)>;

  static Factory &instance() {
    static Factory instance;
    return instance;
  }

  bool registerCreator(const std::string &className, Creator creator) {
    if (!creator) {
      throw std::runtime_error("Cannot register a null creator");
    }

    auto [it, success] =
        creatorRegistry.insert({className, std::move(creator)});
    return success;
  }

  std::shared_ptr<BaseClass>
  create(const std::string &className,
         const ConstructorParams &params = {}) const {
    auto it = creatorRegistry.find(className);
    if (it == creatorRegistry.end()) {
      throw std::runtime_error("Factory error: Class '" + className +
                               "' not registered for base type '" +
                               typeid(BaseClass).name() + "'.");
    }

    try {
      return it->second(params);
    } catch (const std::exception &e) {
      throw std::runtime_error("Factory error: Failed to create '" + className +
                               "': " + e.what());
    }
  }

  bool isRegistered(const std::string &className) const {
    return creatorRegistry.count(className);
  }

private:
  Factory() = default;
  ~Factory() = default;

  // singleton access
  Factory(const Factory &) = delete;
  Factory &operator=(const Factory &) = delete;
  Factory(Factory &&) = delete;
  Factory &operator=(Factory &&) = delete;

  std::map<std::string, Creator> creatorRegistry;
};

} // namespace oop::factory

#endif
