#!/usr/bin/python

from protobuf_to_dict import protobuf_to_dict


fname = "/usr/local/google/home/wojciechz/data/kv/tokens.sst-00000-of-00010"

f = open(fname, "rb")
try:
  byte = f.read(1)
  print "a"
  while byte != "":
    byte = f.read(1)
finally:
  f.close()

print "b"
my_message = MyMessage()
my_message.ParseFromString(pb_my_message)
print protobuf_to_dict(my_message)
