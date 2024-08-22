#include <iostream>
#include <string>
#include <vector>

template <typename T> using myVec = std::vector<T, std::allocator<T>>;

template <typename Container> void test_func(Container container) {
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

int main() {
  // Test test_func with std::vector<int>
  std::vector<int> vec_int;
  test_func(vec_int);
  std::cout << "vec_int size after test_func: " << vec_int.size() << std::endl;

  // Test test_func with myVec<std::string>
  myVec<std::string> vec_string;
  test_func(vec_string);
  std::cout << "vec_string size after test_func: " << vec_string.size()
            << std::endl;

  // Test TemplateTemplateParameter with std::vector<int>
  TemplateTemplateParameter<int, std::vector> ttp_int_vector;

  // Test TemplateTemplateParameter with myVec<std::string>
  TemplateTemplateParameter<std::string, myVec> ttp_string_myvec;

  return 0;
}