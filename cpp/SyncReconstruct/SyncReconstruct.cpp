/*
This file is part of Time Tagger software defined digital data acquisition.

Copyright (C) 2011-2020 Swabian Instruments
All Rights Reserved

Unauthorized copying of this file is strictly prohibited.
*/

#include "SyncReconstruct.h"
#include "TimeTagger.h"
#include <cmath>
#include <iostream>

SyncReconstruct::SyncReconstruct(TimeTaggerBase *tagger, channel_t channel_sync,
                                 double sync_frequency,
                                 std::vector<channel_t> channels_detectors)
    : _Iterator(tagger), channel_sync(channel_sync),
      sync_frequency(sync_frequency), channels_detectors(channels_detectors),
      output_channel(getNewVirtualChannel()) {
  for (auto &elem : channels_detectors)
    registerChannel(elem);
  registerChannel(channel_sync);

  sync_period_in_ps = 1e12 / sync_frequency;

  clear_impl();

  start();
}

SyncReconstruct::~SyncReconstruct() { stop(); }

void SyncReconstruct::clear_impl() {
  delayed_tags.clear();
  last_sync_timestamp = -1;
}

channel_t SyncReconstruct::getChannel() { return output_channel; }

int SyncReconstruct::getFrequency() {
  lock();
  auto result = sync_frequency;
  unlock();
  return result;
}

bool SyncReconstruct::next_impl(std::vector<Tag> &incoming_tags,
                                timestamp_t begin_time, timestamp_t end_time) {
  modified_tag_stream.clear();

  for (const Tag &tag : incoming_tags) {

    delayed_tags.push_back(tag);

    if (tag.type != Tag::Type::TimeTag) {
      // the following is redundant - but I don't know how to make a function
      // out of it
      //
      // do not flush the TimeTag events queued of the detector channels,
      // because otherwise not only signal<->sync pairs are within the stream
      for (const Tag &tag : delayed_tags) {
        bool insert = true;
        if (tag.type == Tag::Type::TimeTag) {
          for (size_t c = 0; c < channels_detectors.size(); c++) {
            insert = false;
            break;
          }
        }
        if (insert) {
          modified_tag_stream.push_back(tag);
        }
      }
      clear_impl();
      continue;
    }

    if (tag.channel == channel_sync) {
      auto sync_timestamp = tag.time;
      // for the linear interpolation, we need to know the timestamp of a
      // previous sync tag arrived
      if (last_sync_timestamp >= 0) {
        auto dt_conditional_sync =
            (uint32_t)(sync_timestamp - last_sync_timestamp);

        double dt_sync_counts_float =
            double(dt_conditional_sync) /
            double(sync_period_in_ps); // this and the next line can be
                                       // optimized, suggestions are welcome
        int sync_counts = std::round(dt_sync_counts_float);

        // check whether the acceptance range of +/- 10%
        if (abs(dt_sync_counts_float - sync_counts) < 0.1) {
          auto dt_sync = dt_conditional_sync / sync_counts;
          // calculate the time stamp of the reconstructed sync
          auto reconstructed_sync_timestamp = sync_timestamp - dt_sync;

          Tag reconstructed_sync_tag = {};
          reconstructed_sync_tag.channel = output_channel;
          reconstructed_sync_tag.type = Tag::Type::TimeTag;
          reconstructed_sync_tag.time = reconstructed_sync_timestamp;
          // insert the reconstructed sync tag
          modified_tag_stream.push_back(reconstructed_sync_tag);
          // flush delayed tags
          for (const Tag &tag : delayed_tags) {
            modified_tag_stream.push_back(tag);
          }
          delayed_tags.clear();
        } else {
          std::cout
              << "\nSync frequency not matching the initial input frequency "
              << (sync_frequency) << "\n\n";
          // the following is redundant - but I don't know how to make a
          // function out of it
          //
          // do not flush the TimeTag events queued of the detector channels,
          // because otherwise not only signal<->sync pairs are within the
          // stream
          for (const Tag &tag : delayed_tags) {
            bool insert = true;
            if (tag.type == Tag::Type::TimeTag) {
              for (size_t c = 0; c < channels_detectors.size(); c++) {
                insert = false;
                break;
              }
            }
            if (insert) {
              modified_tag_stream.push_back(tag);
            }
          }
          clear_impl();
          continue;
        }
      }
      last_sync_timestamp = sync_timestamp;
    }
  }

  // this measurement class is not 100% conform to the requirements of Time
  // Tagger measurements the begin_time and end_time is wrong for the next
  // method for the following measurements, because don't take it into account
  // when we insert the time tags. the solution would be to delay all incoming
  // tags accordingly (> 1x sync period)
  incoming_tags.swap(modified_tag_stream);

  return true;
}
