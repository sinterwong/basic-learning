#include <gtest/gtest.h>
#include <vector>

template <typename T> using myVec = std::vector<T, std::allocator<T>>;

template <typename Container> void test_func(Container &container) {
  typedef
      typename std::iterator_traits<typename Container::iterator>::value_type
          Valtype;
  for (int i = 0; i < 10; i++) {
    container.insert(container.end(), Valtype());
  }
}

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
};

TEST(TestAliasTemplate, TestAliasTemplate) {
  myVec<int> vec;
  test_func(vec);
  ASSERT_EQ(vec.size(), 10);
}

TEST(TestAliasTemplate, TestTemplateTemplateParameter) {
  // Test TemplateTemplateParameter with std::vector<int>
  TemplateTemplateParameter<int, std::vector> ttp_int_vector;

  // Test TemplateTemplateParameter with myVec<std::string>
  TemplateTemplateParameter<std::string, myVec> ttp_string_myvec;
}
