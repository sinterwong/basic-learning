#include "features/perfect_forward.hpp"
#include <algorithm>
using namespace features::forward;

int main()
{
    int a = 10;
    func(a);
    func(std::move(a));

    return 0;
}