#include "SingTag.h"

#include <chrono>
#include <iostream>

SingTag::SingTag(TimeTagger *t, std::vector<channel_t> channels, int outmode)
    : IteratorBase(t), channels(channels), outmode(outmode) {

  for (auto &elem : channels)
    registerChannel(elem);

  clear();
  start();
}

SingTag::~SingTag() { stop(); }

void SingTag::clear_impl() {
  curtag.hex = 0;
  time_offset = 0;
}

bool SingTag::next_impl(std::vector<Tag> &list, timestamp_t begin_time,
                        timestamp_t end_time) {
  for (const Tag &tag : list) {
    if (time_offset == 0) {
      auto now = std::chrono::system_clock::now().time_since_epoch();
      auto nanoseconds =
          std::chrono::duration_cast<std::chrono::nanoseconds>(now);

      time_offset = nanoseconds.count() * 8 - tag.time / 125;
    }

    uint64_t t = (tag.time / 125 + time_offset) & ((1ULL << 49) - 1);

    if (tag.type != Tag::Type::Normal) {
      flush();
      curtag.time = t;
      curtag.un = 0x01;
      time_offset = 0;
      flush();
    }

    for (size_t c = 0; c < channels.size(); c++) {
      if (channels[c] == tag.channel) {
        if (t != curtag.time)
          flush();

        curtag.time = t;
        curtag.bits |= 1 << c;
      }
    }
  }
  return false;
}

void SingTag::flush() {
  if (!curtag.hex)
    return;

  switch (outmode) {
  case 0:
    printf("%08x\n%08x\n", curtag.dv, curtag.cv);
    break;
  case 1:
    fwrite(&curtag.dv, 4, 1, stdout);
    fwrite(&curtag.cv, 4, 1, stdout);
    break;
  case 2:
    printf("%08x%08x\n", curtag.dv, curtag.cv);
    break;
  }

  curtag.hex = 0;
}
