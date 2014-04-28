import numpy as np
from sortedcontainers import SortedSet
import cPickle

sets = ['train', 'valid', 'test']
for name in sets:
  fname = "/Users/wojto/data/pennchr/" + name
  with open(fname + ".txt") as f: 
    text = f.readlines()

  print len(text)
  m = 0
  s = SortedSet()
  for i in xrange(len(text)):
    m = max(m, len(text[i]))
    s = s.union(SortedSet(text[i]))

  cmap = {}
  for i in xrange(len(s)):
    cmap[s[i]] = i

  x = np.zeros((m / 2 + 1, len(text) - 1), dtype=np.uint8)
  for i in xrange(len(text) - 1):
    for j in xrange(len(text[i]) / 2):
      c = text[i][2 * j + 1]
      x[j, i] = cmap[c]
  print np.sum(x, axis=0)
  assert (np.sum(x, axis=0) != 0).all()

  cPickle.dump(x, open(fname + ".pkl", "wb" ))
