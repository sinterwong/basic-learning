#include "logger/logger.hpp"
// #include "threadsafe_queue.hpp"
#include "threadsafe_queue_fg.hpp"
#include <iostream>
#include <thread>
#include <vector>

using namespace concurrency;

int main() {

  BasicLearningLoggerInit(true, true, true, true);

  // threadsafe_queue<int> myQueue;
  threadsafe_queue_fg_done<int> myQueue;

  std::thread t1([&myQueue]() {
    for (size_t i = 0; i < 10; ++i) {
      myQueue.push(i);
    }
  });

  std::thread t2([&myQueue]() {
    for (size_t i = 0; i < 3; ++i) {
      BASIC_LOGGER_INFO("waitPopData: {}", *myQueue.wait_and_pop());
    }
  });

  std::thread t3([&myQueue]() {
    for (size_t i = 0; i < 7; ++i) {
      BASIC_LOGGER_INFO("tryPopData: {}", *myQueue.try_pop());
    }
  });

  t1.join();
  t2.join();
  t3.join();

  BasicLearningLoggerDrop();

  return 0;
}