/*
This file is part of Time Tagger software defined digital data acquisition.

Copyright (C) 2011-2020 Swabian Instruments
All Rights Reserved

Unauthorized copying of this file is strictly prohibited.
*/
#pragma once

#include "TimeTagger.h"

#include <fstream>
#include <iostream>
#include <vector>

/*
 *   Binary format: 64 bit, 32 bit, 64 bit
 */
struct Output {
  // sync number
  int64_t sync_no;

  // the channel number
  channel_t channel;

  // time difference to the last sync
  timestamp_t dt_from_sync;
};

/*
 *   Example to explain how to create a custom measurement class
 */
class TimeDifferencesDumper : public _Iterator {
public:
  // constructor
  TimeDifferencesDumper(TimeTaggerBase *tagger,
                        channel_t channel_reconstructed_sync,
                        double sync_frequency,
                        std::vector<channel_t> channels_detectors,
                        std::string output_filename, bool binary_output);

  ~TimeDifferencesDumper();

  int64_t getNumberOfDumpedEvents();

protected:
  /*
   * next recieves and handles the incoming tags
   */
  bool next_impl(std::vector<Tag> &incoming_tags, timestamp_t begin_time,
                 timestamp_t end_time) override;

  /*
   * reset measurement
   */
  void clear_impl() override;

  /*
   * callback before the measurement is started
   */
  void on_start() override;

  /*
   * callback after the measurement is stopped
   */
  void on_stop() override;

private:
  // channels used for this measurement
  const std::vector<channel_t> channels_detectors;
  channel_t channel_sync;
  int64_t sync_counter;
  timestamp_t last_sync_timestamp;
  int64_t dumped_events_counter;
  std::ofstream output_file;
  std::string filename;
  bool binary_output;
  std::vector<Output> listDump;
  double sync_frequency;
  timestamp_t sync_period_in_ps;
};
