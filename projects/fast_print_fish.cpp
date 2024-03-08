#include <condition_variable>
#include <iostream>
#include <mutex>
#include <ostream>
#include <thread>
#include <unordered_map>

enum State { A = 1, B, C, D, E, F };

int currentState = A;
std::mutex lk;

struct Rule {
  int from;
  char op;

  Rule(int from, char op) : from(from), op(op) {}
  bool operator==(const Rule& other) const {
    // std::cout << "== called\n\n\n" << std::endl;
    // return from == other.from && op == other.op;
    return true;
    // perfect hash, one bucket one key. Always true
  }
};

namespace std {
template <>
struct hash<Rule> {
  size_t operator()(const Rule& p) const {
    return static_cast<size_t>(p.from) * 3 +
           (p.op == '<' ? 0 : (p.op == '>' ? 1 : 2));
  }
};
}  // namespace std

std::unordered_map<Rule, int> stateTransferMap;

void initTransferMap() {
  stateTransferMap[Rule(A, '<')] = B;
  stateTransferMap[Rule(B, '>')] = D;
  stateTransferMap[Rule(D, '<')] = E;
  stateTransferMap[Rule(A, '>')] = C;
  stateTransferMap[Rule(C, '<')] = F;
  stateTransferMap[Rule(F, '>')] = E;
  stateTransferMap[Rule(E, ' ')] = A;
}

int next(char ch) {
  auto key = stateTransferMap.find(Rule(currentState, ch));
  if (key != stateTransferMap.end()) {
    currentState = key->second;
    return key->second;
  }
  return 0;
}

bool inline canPrint(char ch) { return next(ch) == 0 ? false : true; }

void printLeft() {
  while (true) {
    lk.lock();
    if (canPrint('<')) {
      std::cout << "<";
      // std::flush(std::cout);
    }
    lk.unlock();
  }
}

void printRight() {
  while (true) {
    lk.lock();
    if (canPrint('>')) {
      std::cout << ">";
      // std::flush(std::cout);
    }
    lk.unlock();
  }
}

void printBlank() {
  while (true) {
    lk.lock();
    if (canPrint(' ')) {
      std::cout << " ";
      // std::flush(std::cout);
    }
    lk.unlock();
  }
}

int main() {
  initTransferMap();

  std::thread t1(printLeft);
  std::thread t2(printRight);
  std::thread t3(printBlank);
  t1.join();
  t2.join();
  t3.join();
}