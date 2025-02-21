#include <httplib.h>

#include <gtest/gtest.h>

TEST(HttplibTest, Normal) {
  httplib::Server svr;
  svr.Get("/hello", [](const auto &req, auto &res) {
    res.set_content("Hello World!", "text/plain");
  });
  svr.stop();
}
