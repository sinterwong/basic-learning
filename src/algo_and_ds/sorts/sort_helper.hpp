

#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iterator>

namespace algo_and_ds {
namespace sort {

template <typename Container> void printArray(Container const &container) {
  for (auto &i : container) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;
}

template <typename ForwardIterator>
bool isSorted(ForwardIterator first, ForwardIterator last) {
  for (auto iter = ++first; iter != last; iter++) {
    if (*iter < *(iter - 1)) {
      return false;
    }
  }
  return true;
}

template <typename Container>
void generateRandomArray(Container &container, int range) {
  using valType =
      typename std::iterator_traits<typename Container::iterator>::value_type;
  srand(time(nullptr));
  for (auto &i : container) {
    i = static_cast<valType>(rand() % range);
  }
}

template <typename Container>
void generateNearlyOrderedArray(Container &container, int num) {
  using valType =
      typename std::iterator_traits<typename Container::iterator>::value_type;
  for (size_t i = 0; i < container.size(); i++) {
    container[i] = static_cast<valType>(i);
  }
  srand(time(nullptr));
  for (int i = 0; i < num; i++) {
    int posx = rand() % container.size();
    int posy = rand() % container.size();
    std::swap(container[posx], container[posy]);
  }
}

template <typename Container>
void testSort(std::string const &name, void (*sort)(Container &),
              Container &container) {
  using namespace std::chrono;
  auto start = steady_clock::now();
  sort(container);
  auto end = steady_clock::now();
  assert(isSorted(container.cbegin(), container.cend()));
  float micro_s = duration_cast<microseconds>(end - start).count();
  std::cout << name << ": " << micro_s / 1000 / 1000
            << "s" << std::endl;
}

} // namespace sort
} // namespace algo_and_ds