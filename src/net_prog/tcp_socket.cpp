/**
 * @file tcp_socket.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "tcp_socket.hpp"
#include <arpa/inet.h>
#include <cstdint>
#include <netinet/in.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

namespace net_prog {
TCPSocket::TCPSocket() : sockfd(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) {
  check_error(sockfd, "Failed to create socket!");
}
TCPSocket::~TCPSocket() {
  if (sockfd != -1) {
    close(sockfd);
  }
}

void TCPSocket::connect(const std::string &ip, uint16_t port) {
  sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);

  check_error(inet_pton(AF_INET, ip.c_str(), &addr.sin_addr),
              "Invalid IP address");

  check_error(
      ::connect(sockfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)),
      "Failed to connect");
}

void TCPSocket::bind(uint16_t port) {
  sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;
  check_error(::bind(sockfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)),
              "Failed to bind");
}

void TCPSocket::listen(int backlog) {
  check_error(::listen(sockfd, backlog), "Failed to listen");
}

TCPSocket TCPSocket::accept() {
  sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);
  int new_fd = ::accept(sockfd, reinterpret_cast<sockaddr *>(&addr), &addr_len);
  check_error(new_fd, "Failed to accept");
  return TCPSocket(new_fd);
}

void TCPSocket::send(std::span<const char> buffer) {
  check_error(::send(sockfd, buffer.data(), buffer.size(), 0),
              "Failed to send");
}

int TCPSocket::recv(std::span<char> buffer) {
  int bytes_received = ::recv(sockfd, buffer.data(), buffer.size(), 0);
  check_error(bytes_received, "Failed to receive");
  return bytes_received;
}

void TCPSocket::check_error(int ret, const std::string &message) {
  if (ret == -1) {
    throw std::runtime_error(message);
  }
}
} // namespace net_prog
