For compiling this example, you need to link it with libTimeTagger.so:

  g++ main.cpp SingTag.cpp -l TimeTagger -o SingTagDumper

Afterwards, it can be run without arguments:
  ./SingTagDumper

This example was tested with the stable upstream version 2.2.4.

Notes:
- The provided TimeTagger.h must match the installed Time Tagger version.
- The UTC estimation is queried with an unknown latency of up to one second.
- The "time / 125" implementation has a broken overflow behavior after 100 days.
