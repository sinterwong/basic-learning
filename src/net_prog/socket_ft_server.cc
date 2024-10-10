/**
 * @file socket_ft_server.cc
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

DEFINE_uint32(port, 9797, "Specify the port.");

namespace fs = std::filesystem;

void receiveFile(net_prog::TCPSocket &socket, const fs::path &filePath) {
  std::ofstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to create file: " << filePath << std::endl;
    return;
  }

  char buffer[4096];
  int bytesReceived;
  while ((bytesReceived = socket.recv(std::span<char>(buffer)))) {
    file.write(buffer, bytesReceived);
  }
  file.close();
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  net_prog::TCPSocket server;
  server.bind(FLAGS_port);
  server.listen(5);
  std::cout << "Server is listening on port " << FLAGS_port << std::endl;

  net_prog::TCPSocket client = server.accept();
  std::cout << "Client connected!" << std::endl;

  receiveFile(client, "received_file.txt");
}