import numpy as np
import cPickle

sets = ['train', 'valid', 'test']
for name in sets:
  fname = "/Users/wojto/data/pennchr/" + name
  with open(fname + ".txt") as f: 
    text = f.readlines()

  print len(text)
  m = 0
  lens = []
  for i in xrange(len(text)):
    lens.append(len(text[i]))

  import ipdb
  ipdb.set_trace()
  exit(0)
  m = max(m, len(text[i]))

  x = 255 * np.ones((m / 2 + 1, len(text) - 1), dtype=np.uint8)
  for i in xrange(len(text) - 1):
    for j in xrange(len(text[i]) / 2):
      c = text[i][2 * j + 1]
      x[j, i] = ord(c)
  print np.sum(x, axis=0)
  assert (np.mean(x, axis=0) != 255).all()

  cPickle.dump(x, open(fname + ".pkl", "wb" ))
