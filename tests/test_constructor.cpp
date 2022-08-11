#include <iostream>

struct A {
  int a;
};

struct B {
  B(){};
  int b;
};

struct C {
  C(int c_) : c(c_) {}
  C() = default;
  int c;
};

int main(int argc, char **argv) {

  A a;
  std::cout << a.a << std::endl;  
  a = A();
  std::cout << a.a << std::endl;  

  B b;
  std::cout << b.b << std::endl;

  C c;
  std::cout << c.c << std::endl;

  return 0;
}
