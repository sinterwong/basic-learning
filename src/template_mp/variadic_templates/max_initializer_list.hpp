#ifndef __FEATURES_MAX_INITIALIZER_LIST_HPP_
#define __FEATURES_MAX_INITIALIZER_LIST_HPP_

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <stdexcept>

namespace template_mp {
namespace variadic_templates {

struct __Iter_less_iter {
  template <typename __Iterator1, typename __Iterator2>
  bool operator()(__Iterator1 it1, __Iterator2 it2) const {
    return *it1 < *it2;
  }
};

inline __Iter_less_iter __iter_less_iter() { return __Iter_less_iter(); }

template <typename _ForwardIterator, typename Compare>
inline _ForwardIterator __max_element(_ForwardIterator __first,
                                      _ForwardIterator __last, Compare __comp) {
  if (__first == __last) {
    return __first;
  }
  _ForwardIterator result = __first;
  while (++__first != __last) {
    if (__comp(result, __first)) {
      result = __first;
    }
  }
  return result;
}

template <typename _ForwardIterator>
inline _ForwardIterator max_element(_ForwardIterator __first,
                                    _ForwardIterator __end) {
  return __max_element(__first, __first, __iter_less_iter);
}

template <typename _Tp> inline _Tp max(std::initializer_list<_Tp> __l) {
  return *max_element(__l.begin(), __l.end());
}

template <typename T> T maximum(T n) { return n; }

template <typename T, typename... Args> T maximum(T n, Args... args) {
  return std::max(n, maximum(args...));
}

} // namespace variadic_templates
} // namespace template_mp

#endif