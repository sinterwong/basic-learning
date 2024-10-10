/**
 * @file socket_ft_client.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "tcp_socket.hpp"
#include <filesystem>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <span>

DEFINE_string(ip, "127.0.0.1", "Specify the ip.");
DEFINE_uint32(port, 9797, "Specify the port.");
DEFINE_string(file, "", "Specify the file.");

namespace fs = std::filesystem;

void sendFile(net_prog::TCPSocket &socket, const fs::path &filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to create file: " << filePath << std::endl;
    return;
  }

  char buffer[4096];
  int bytesReceived;
  while (file.read(buffer, sizeof(buffer))) {
    socket.send(std::span<const char>(buffer));
  }
  // Send remaining data after the last full buffer
  socket.send({buffer, static_cast<size_t>(file.gcount())});
  file.close();
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  net_prog::TCPSocket client;
  client.connect(FLAGS_ip, FLAGS_port);
  std::cout << "Connected to server!" << std::endl;
  sendFile(client, FLAGS_file);
}