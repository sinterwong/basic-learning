#include "dummy.hpp"
#include <logger.hpp>

namespace bl::dummy {
void MyDummy::doSomething() { LOG_INFOS << "Hello World!"; }

} // namespace bl::dummy