#!/usr/bin/python
import numpy as np
import cPickle
from math import ceil

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

  order = sorted(zip(lens, xrange(len(lens))))
  data = []
  order = order[1:]
  bs = 10
  for mb in xrange(int(ceil(len(text) / bs))):
    subord = order[mb*bs:min((mb+1)*bs, len(text))]
    x = 255 * np.ones((subord[-1][0] / 2, len(subord)), dtype=np.uint8)
    for i in xrange(len(subord)):
      for j in xrange(subord[i][0] / 2):
        c = text[subord[i][1]][2 * j + 1]
        if c == '_':
          c = ' '
        x[j, i] = ord(c)
    data.append(x)

  cPickle.dump(data, open(fname + ".pkl", "wb" ))
