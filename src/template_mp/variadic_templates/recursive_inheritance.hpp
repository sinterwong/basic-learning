#ifndef __FEATURES_RECURIVE_INHERITANCE_HPP_
#define __FEATURES_RECURIVE_INHERITANCE_HPP_

namespace template_mp {
namespace variadic_templates {

template <typename... Values> class tuple {};

template <> class tuple<> {};

template <typename Head, typename... Tails>
class tuple<Head, Tails...> : private tuple<Tails...> {
  using inheritance = tuple<Tails...>;

public:
  tuple() {}
  tuple(Head v, Tails... vtails) : m_head(v), inheritance(vtails...) {}

  Head head() { return m_head; }

  // 此处返回基类的类型，但是本类型的指针，会发生截断，因此获取的就是基类
  inheritance &tail() { return *this; };

private:
  Head m_head;
};

} // namespace variadic_templates
} // namespace template_mp

#endif