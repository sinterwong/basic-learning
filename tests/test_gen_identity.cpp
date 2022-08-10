#include <iostream>
#include <chrono>
#include <sstream>
#include <random>
#include <string>

using std::chrono::milliseconds;

unsigned int random_char() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    return dis(gen);
}

std::string generate_hex(const unsigned int len) {
    std::stringstream ss;
    for (auto i = 0; i < len; i++) {
        const auto rc = random_char();
        std::stringstream hexstream;
        hexstream << std::hex << rc;
        auto hex = hexstream.str();
        ss << (hex.length() < 2 ? '0' + hex : hex);
    }
    return ss.str();
}


int main(int argc, char ** argv) {
  std::string identity = generate_hex(16);
  std::cout << "identity: " << identity << std::endl;
  return 0;
}
