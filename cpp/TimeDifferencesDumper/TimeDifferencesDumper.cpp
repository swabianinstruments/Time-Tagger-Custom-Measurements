/*
This file is part of Time Tagger software defined digital data acquisition.

Copyright (C) 2011-2020 Swabian Instruments
All Rights Reserved

Unauthorized copying of this file is strictly prohibited.
*/

#include "TimeDifferencesDumper.h"
#include "TimeTagger.h"
#include <cmath>

#include <cstddef>

TimeDifferencesDumper::TimeDifferencesDumper(
    TimeTaggerBase *tagger, channel_t channel_sync, double sync_frequency,
    std::vector<channel_t> channels_detectors, std::string output_filename,
    bool binary_output)
    : _Iterator(tagger), channels_detectors(channels_detectors),
      channel_sync(channel_sync), filename(output_filename),
      binary_output(binary_output), sync_frequency(sync_frequency) {
  // we have to tell the FPGA which channels are currently in use - so register
  // them (tags of channels which are not used are not sent to the computer)
  for (auto &elem : channels_detectors)
    registerChannel(elem);
  registerChannel(channel_sync);

  sync_period_in_ps = 1e12 / sync_frequency;

  sync_counter = 0;
  dumped_events_counter = 0;

  clear_impl();

  start();
}

TimeDifferencesDumper::~TimeDifferencesDumper() { stop(); }

int64_t TimeDifferencesDumper::getNumberOfDumpedEvents() {
  lock();
  auto result = dumped_events_counter;
  unlock();

  return result;
}

void TimeDifferencesDumper::clear_impl() { last_sync_timestamp = -1; }

void TimeDifferencesDumper::on_start() {
  if (binary_output) {
    output_file.open(filename.c_str(),
                     std::ios::out | std::ios::trunc | std::ios::binary);
  } else {
    std::cout << "\nWARNING: Plain text output file is inefficient and will "
                 "might lead to overflows. Please use the binary output format "
                 "(can be changed via setting binary_output_file = true).\n\n";
    output_file.open(filename.c_str(),
                     std::ios::out | std::ios::trunc | std::ios::binary);
  }
#ifdef _WIN32
  output_file.rdbuf()->pubsetbuf(nullptr, 1024 * 1024);
#endif
}
void TimeDifferencesDumper::on_stop() { output_file.close(); }

// here we handle the incoming time tags
bool TimeDifferencesDumper::next_impl(std::vector<Tag> &incoming_tags,
                                      timestamp_t begin_time,
                                      timestamp_t end_time) {
  // iterate over all the tags recievied
  listDump.clear();
  for (const Tag &tag : incoming_tags) {
    if (tag.type != Tag::Type::TimeTag) {
      // you always have to handle the overflows.
      // here you must implement what should happen when FPGA data buffer is
      // overflowing because of the limited transfer bandwidth sometimes you
      // have to restart the measurement - sometimes you're fine with the amount
      // of missed events - sometimes you don't care about overflows.
      last_sync_timestamp = -1;
      continue;
    }

    if (tag.channel == channel_sync) {
      auto sync_timestamp = tag.time;
      auto dt_sync = sync_timestamp - last_sync_timestamp;
      if (last_sync_timestamp >= 0) {

        double dt_sync_counts_float =
            dt_sync / sync_period_in_ps; // this and the next line can be
                                         // optimized, suggestions are welcome
        int sync_counts = std::round(dt_sync_counts_float);

        sync_counter += sync_counts;
      } else {
        last_sync_timestamp = sync_timestamp;
        sync_counter++; // in case of an overflow the sync counter is not
                        // increased as it should - this should be improved
      }
      last_sync_timestamp = sync_timestamp;
    } else {
      if (last_sync_timestamp >= 0) {
        for (size_t c = 0; c < channels_detectors.size(); c++) {
          if (tag.channel == channels_detectors[c]) {
            Output output = {};
            output.sync_no = sync_counter;
            output.channel = tag.channel;
            output.dt_from_sync = tag.time - last_sync_timestamp;
            listDump.push_back(output);
          }
        }
      }
    }
  }

  int64_t copyElements = listDump.size();
  if (binary_output) {
    output_file.write((char *)listDump.data(), copyElements * sizeof(Output));
  } else {
    for (auto const &data : listDump) {
      output_file << data.sync_no << " " << data.channel << " "
                  << data.dt_from_sync << "\n";
    }
  }
  dumped_events_counter += copyElements;

  // return if incoming_tags was modified. If so, please keep care about the
  // requirements:
  // -- all tags must be sorted
  // -- begin_time <= tags < end_time
  return false;
}
