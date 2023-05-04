#include <deque>
#include <future>
#include <iostream>
#include <mutex>

std::deque<std::packaged_task<void()>> tasks;
std::mutex m;

bool gui_shutdown_message_received() { return false; }

void get_and_process_gui_message() { std::cout << "Hello gui" << std::endl; }

void gui_thread() {

  while (!gui_shutdown_message_received()) { // 轮询任务
    get_and_process_gui_message();
    std::packaged_task<void()> task;
    {
      std::lock_guard<std::mutex> lk(m);
      if (tasks.empty()) {
        continue;
      }
      task = std::move(tasks.front());
      tasks.pop_front();
    }
    task(); // 执行任务
  }
}

std::thread gui_bg_thread(gui_thread);

template <typename Func> std::future<void> post_task_for_gui_thread(Func f) {
  std::packaged_task<void()> task(f);
  std::future<void> res = task.get_future();
  std::lock_guard<std::mutex> lk(m);
  tasks.push_back(std::move(task));
  return res;
}

int main() { return 0; }