/**
 * @file tcp_socket.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __NETWORK_PROGRAMING_TCP_SOCKET_HPP_
#define __NETWORK_PROGRAMING_TCP_SOCKET_HPP_

#include <cstdint>
#include <span>
#include <string>
namespace net_prog {

class TCPSocket {
public:
  TCPSocket();
  ~TCPSocket();

  void connect(const std::string &ip, uint16_t port);

  void bind(uint16_t port);

  void listen(int backlog);

  TCPSocket accept();

  void send(std::span<const char> buffer);

  int recv(std::span<char> buffer);

  void check_error(int ret, const std::string &message);

private:
  TCPSocket(int fd) : sockfd(fd) {}

  int sockfd;
};
} // namespace net_prog

#endif