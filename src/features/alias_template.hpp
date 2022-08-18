/**
 * @file alias_template.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @version 0.1
 * @date 2022-08-18
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __FEATURES_ALIAS_TEMPLATE_HPP_
#define __FEATURES_ALIAS_TEMPLATE_HPP_

#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

namespace features {
namespace alias_templates {

template <typename T> using myVec = std::vector<T, std::allocator<T>>;

template <typename Container> void test_func(Container container) {

  /**
   * @brief
   * 利用container的iterator，通过iterator_traits萃取出container中value的类型,
   * 在给一个typedef
   *
   */
  typedef typename std::iterator_traits<typename Container::iterator>::val_type
      Valtype;
  for (int i = 0; i < 10; i++) {
    container.insert(container.end(), Valtype());
  }
  // Error: c++ 只能传递对象不能传递类型
  // test_func(container, elemType);

  // Succese: 通过传入的对象可能获取container的iterator,
  // 通过iterator可以获取value_type test_func(container<T>())
}

/**
 * @brief 模板模板参数, 语法挺奇怪
 *
 * @tparam T value type
 * @tparam Container 容器类型，且容器是一个模板类
 */
template <typename T, template <typename> typename Container>
class TemplateTemplateParameter {
private:
  Container<T> c;

public:
  TemplateTemplateParameter() {
    for (int i = 0; i < 10; i++) {
      c.insert(c.end(), T());
    }
  }

  // TemplateTemplateParameter<std::string, myVec> ttp;  // vector<>
  // 模板有两个参数，因此需要自己设置一个alisa
};

} // namespace alias_templates
} // namespace features

#endif