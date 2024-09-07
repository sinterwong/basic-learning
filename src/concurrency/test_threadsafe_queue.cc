#include "logger/logger.hpp"
// #include "threadsafe_queue.hpp"
#include "threadsafe_list.hpp"
#include "threadsafe_lookup_table.hpp"
#include "threadsafe_queue_fg.hpp"
#include <iostream>
#include <thread>
#include <vector>

using namespace my_concurrency;

int main() {

  BasicLearningLoggerInit(true, true, true, true);

  // threadsafe_queue<int> myQueue;
  threadsafe_queue_fg_done<int> myQueue;

  std::thread t0([&myQueue]() {
    while (true) {
      myQueue.push(rand());
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });

  std::thread t1([&myQueue]() {
    while (true) {
      int temp = rand();
      myQueue.push(temp);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });

  std::thread t2([&myQueue]() {
    for (;;) {
      auto v_ptr = myQueue.wait_and_pop();
      if (v_ptr) {
        BASIC_LOGGER_INFO("waitPopData: {}", *v_ptr);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
      }
    }
  });

  std::thread t3([&myQueue]() {
    for (;;) {
      auto v_ptr = myQueue.try_pop();
      if (v_ptr) {
        BASIC_LOGGER_INFO("tryPopData: {}", *v_ptr);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
      }
    }
  });

  t0.join();
  t1.join();
  t2.join();
  t3.join();

  BasicLearningLoggerDrop();

  return 0;
}