#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

namespace my_concurrency {
class GuiThread {
public:
  GuiThread() : shutdown_requested_(false) {
    gui_thread_ = std::thread(&GuiThread::gui_thread_func, this);
  }

  ~GuiThread() {
    shutdown();
    if (gui_thread_.joinable()) {
      gui_thread_.join();
    }
  }

  template <typename Func> std::future<void> post_task(Func f) {
    std::packaged_task<void()> task(f);
    std::future<void> res = task.get_future();
    {
      std::lock_guard<std::mutex> lk(m_);
      tasks_.push_back(std::move(task));
    }
    return res;
  }

  void shutdown() {
    post_task([this]() { shutdown_requested_ = true; });
  }

private:
  void gui_thread_func() {
    while (!shutdown_requested_) {
      get_and_process_gui_message();
      std::packaged_task<void()> task;
      {
        std::lock_guard<std::mutex> lk(m_);
        if (tasks_.empty()) {
          continue;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      if (task.valid()) { // 防止取出来一个空的task
        task();
      }
    }
    std::cout << "GUI thread exiting..." << std::endl;
  }

  void get_and_process_gui_message() { std::cout << "Hello gui" << std::endl; }

  std::thread gui_thread_;
  std::mutex m_;
  std::deque<std::packaged_task<void()>> tasks_;
  bool shutdown_requested_;
};
} // namespace my_concurrency