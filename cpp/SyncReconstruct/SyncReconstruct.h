/*
This file is part of Time Tagger software defined digital data acquisition.

Copyright (C) 2011-2020 Swabian Instruments
All Rights Reserved

Unauthorized copying of this file is strictly prohibited.
*/

#pragma once

#include "TimeTagger.h"
#include <vector>

/**
 * \ingroup ITERATOR
 *
 * \brief Add the sync event before the signal event to a virutal channel.
 *
 * With the conditional filter, only the sync event after a signal event is
 * avaiable on the software side. SyncReconstruct inserts the corresponding sync
 * tags before the event into a virtual channel via lineral interpolation.
 * Signal events witout a sync signal, caused e.g. via an overflow, are deleted.
 */

class SyncReconstruct : public _Iterator {
public:
  /**
   * \brief constructor of a SyncReconstruct
   *
   * \param tagger              reference to a TimeTagger
   * \param channel_sync        sync signal channel - must be the triggered
   * channel of the conditional filter \param sync_frequency      frequency of
   * the sync signal (please measure and insert via Countrate) \param
   * channels_detectors  detector event channels - must be the trigger channels
   * of the conditional filter
   */
  SyncReconstruct(TimeTaggerBase *tagger, channel_t channel_sync,
                  double sync_frequency,
                  std::vector<channel_t> channels_detectors);

  ~SyncReconstruct();

  channel_t getChannel();

  int getFrequency();

protected:
  bool next_impl(std::vector<Tag> &incoming_tags, timestamp_t begin_time,
                 timestamp_t end_time) override;
  void clear_impl() override;

private:
  std::vector<Tag> modified_tag_stream; // Tag mirror for injecting new tags
  std::vector<Tag> delayed_tags;        // delay the tags until a sync arrives

  const std::vector<channel_t> channels_detectors;
  channel_t channel_sync;

  const double sync_frequency;
  const channel_t output_channel;
  timestamp_t last_sync_timestamp;
  timestamp_t sync_period_in_ps;
};
