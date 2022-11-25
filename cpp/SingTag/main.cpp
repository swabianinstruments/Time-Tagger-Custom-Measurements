#include <chrono>
#include <thread>

#include "SingTag.h"
#include "TimeTagger.h"

int main() {
  TimeTagger *t = createTimeTagger();
  std::vector<channel_t> v = {1, 2, 3, 4, 5, 6, 7, 8};
  SingTag sing_tag(t, v, 2);
  sing_tag.startFor(100e12);
  while (sing_tag.isRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  freeTimeTagger(t);
}
