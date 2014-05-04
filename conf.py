from os.path import expanduser
import socket

DATA_DIR = "%s/data/" % (expanduser("~"))
if "cs.nyu.edu" in socket.gethostname():
  DUMP_DIR = "/scratch/zaremba/dump/"
else:
  DUMP_DIR = "%s/dump/" % (expanduser("~"))
