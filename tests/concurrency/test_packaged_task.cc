#include "gui_packaged_task.hpp"
#include <future>
#include <gtest/gtest.h>
#include <iostream>

using namespace my_concurrency;

TEST(GuiThreadTest, NormalExecution) {
  GuiThread gui_thread;
  auto f = gui_thread.post_task(
      []() { std::cout << "Hello from packaged_task" << std::endl; });
  f.get();
  // Output should be:
  // Hello gui
  // Hello from packaged_task
}

TEST(GuiThreadTest, MultipleTasks) {
  GuiThread gui_thread;
  std::vector<std::future<void>> futures;
  for (int i = 0; i < 5; ++i) {
    futures.push_back(gui_thread.post_task(
        [i]() { std::cout << "Task " << i << " executed" << std::endl; }));
  }
  for (auto &f : futures) {
    f.get();
  }
  // Output should be (order might vary):
  // Hello gui
  // Task 0 executed
  // Hello gui
  // Task 1 executed
  // Hello gui
  // Task 2 executed
  // Hello gui
  // Task 3 executed
  // Hello gui
  // Task 4 executed
}

TEST(GuiThreadTest, Shutdown) {
  {
    GuiThread gui_thread;
    gui_thread.post_task(
        []() { std::cout << "Task before shutdown" << std::endl; });
    gui_thread.shutdown();
    // Output should be:
    // Hello gui
    // Task before shutdown
    // Hello gui
    // GUI thread exiting...
  }
}

TEST(GuiThreadTest, ShutdownWithPendingTasks) {
  {
    GuiThread gui_thread;
    gui_thread.post_task(
        []() { std::cout << "Task before shutdown" << std::endl; });
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 5; ++i) {
      futures.push_back(gui_thread.post_task(
          [i]() { std::cout << "Task " << i << " executed" << std::endl; }));
    }
    gui_thread.shutdown();
    for (auto &f : futures) {
      f.wait(); // 防止future析构crash
    }
    // Output should be:
    // Hello gui
    // Task before shutdown
    // Hello gui
    // GUI thread exiting...
    // Task 0/1/2/3/4 executed
    // 可能执行，也可能不执行，取决于任务有没有开始执行
  }
}

TEST(GuiThreadTest, PostTaskAfterShutdown) {
  GuiThread gui_thread;
  gui_thread.shutdown();
  auto f = gui_thread.post_task(
      []() { std::cout << "Task after shutdown" << std::endl; });
  // wait_for 等待一段时间, 因为任务已经终结，f.wait()会永远等待下去
  auto status = f.wait_for(std::chrono::milliseconds(100));
  EXPECT_EQ(status, std::future_status::timeout);
  // Output should be:
  // Hello gui
  // GUI thread exiting...
  // "Task after shutdown" should not be printed.
}