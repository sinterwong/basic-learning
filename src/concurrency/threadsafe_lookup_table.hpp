#ifndef _BASIC_CONCUNNENCY_THREADSAFE_LOOKUP_TABLE_HPP_
#define _BASIC_CONCUNNENCY_THREADSAFE_LOOKUP_TABLE_HPP_

#include <algorithm>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace concurrency {
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class threadsafe_lookup_table {
private:
  class bucket_type {
    using bucket_value = std::pair<Key, Value>;
    using bucket_data = std::list<bucket_value>;
    using bucket_iterator = typename std::list<bucket_value>::iterator;

    bucket_data data; // 桶
    std::shared_mutex mutex;

    bucket_iterator find_entry_for(Key const &key) const {
      return std::find_if(
          data.begin(), data.end(),
          [&](bucket_value const &item) { return item.first == key; });
    }

  public:
    Value value_for(Key const &key, Value const &default_value) const {
      std::shared_lock slk(mutex);
      bucket_iterator const found_entry = find_entry_for(key);
      return found_entry == data.end() ? default_value : found_entry->second;
    }

    void add_or_update_mapping(Key const &key, Value const &value) {
      std::unique_lock lock(mutex);
      bucket_iterator const found_entry = find_entry_for(key);
      if (found_entry == data.end()) {
        data.push_back(bucket_value(key, value));
      } else {
        found_entry->second = value;
      }
    }

    void remove_mapping(Key const &key) {
      std::unique_lock lock(mutex);
      bucket_iterator const found_entry = find_entry_for(key);
      if (found_entry != data.end()) {
        data.erase(found_entry);
      }
    }
  };

  std::vector<std::unique_ptr<bucket_type>> buckets;
  Hash hasher;
  bucket_type &get_bucket(Key const &key) const {
    std::size_t const bucket_index = hasher(key) % buckets.size();
    return *buckets.at(bucket_index);
  }

public:
  using key_type = Key;
  using mapped_type = Value;
  using hash_type = Hash;
  threadsafe_lookup_table(size_t num_buckets = 19, Hash const &hasher_ = Hash())
      : buckets(num_buckets), hasher(hasher_) {
    for (size_t i = 0; i < buckets.size(); ++i) {
      buckets.at(i).reset(std::make_unique<bucket_type>());
    }
  }

  threadsafe_lookup_table(threadsafe_lookup_table const &) = delete;
  threadsafe_lookup_table &operator=(threadsafe_lookup_table const &) = delete;

  Value value_for(Key const &key, Value const &default_value = Value()) const {
    return get_bucket(key).value_for(key, default_value);
  }

  void add_or_update_mapping(Key const &key, Value const &value) {
    get_bucket(key).add_or_update_mapping(key, value);
  }

  void remove_mapping(Key const &key) { get_bucket(key).remove_mapping(key); }
};

} // namespace concurrency

#endif
