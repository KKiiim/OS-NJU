#include <condition_variable>
#include <iostream>
#include <mutex>
#include <ostream>
#include <thread>
#include <vector>

enum State { A = 1, B, C, D, E, F };

int currentState = A;
std::mutex lk;
std::condition_variable cv;

struct Rule {
  int from;
  char op;
  int to;
};
// clang-format off
std::vector<Rule> stateTransfer = {
    {A, '<', B}, 
    {B, '>', D}, 
    {D, '<', E}, 
    {A, '>', C},
    {C, '<', F}, 
    {F, '>', E}, 
    {E, ' ', A},
};
// clang-format on

int next(char ch) {
  for (auto it : stateTransfer) {
    if (it.from == currentState && it.op == ch) {
      currentState = it.to;
      return it.to;
    }
  }
  return 0;
}

bool inline canPrint(char ch) { return next(ch) == 0 ? false : true; }

void printLeft() {
  while (true) {
    lk.lock();
    if (canPrint('<')) {
      std::cout << "<";
      std::flush(std::cout);
    }
    lk.unlock();
  }
}

void printRight() {
  while (true) {
    lk.lock();
    if (canPrint('>')) {
      std::cout << ">";
      std::flush(std::cout);
    }
    lk.unlock();
  }
}

void printBlank() {
  while (true) {
    lk.lock();
    if (canPrint(' ')) {
      std::cout << " ";
      std::flush(std::cout);
    }
    lk.unlock();
  }
}

int main() {
  std::thread t1(printLeft);
  std::thread t2(printRight);
  std::thread t3(printBlank);
  t1.join();
  t2.join();
  t3.join();
}