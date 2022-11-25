#pragma once

#include "TimeTagger.h"

class SingTag : public IteratorBase {
public:
  SingTag(TimeTagger *t, std::vector<channel_t> channels, int outmode = 0);

  ~SingTag();

protected:
  bool next_impl(std::vector<Tag> &incoming_tags, timestamp_t begin_time,
                 timestamp_t end_time) override;

  void clear_impl() override;

private:
  void flush();

  std::vector<channel_t> channels;
  timestamp_t time_offset = 0;

  int outmode;

  union singtag {
    struct {
      uint64_t bits : 8;
      uint64_t un : 7;
      uint64_t time : 49;
    };
    struct {
      uint32_t cv;
      uint32_t dv;
    };
    uint64_t hex;

  } curtag;
};
