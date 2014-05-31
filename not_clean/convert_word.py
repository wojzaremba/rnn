#!/usr/bin/python
import numpy as np
import cPickle
from math import ceil

sets = ['train', 'valid', 'test']
words = set()
path = "/usr/local/google/home/wojciechz/data/pennword/"
for name in sets:
  fname = path + "ptb." + name
  with open(fname + ".txt") as f: 
    text = f.readlines()
  for tt in text:
    tt = tt.split(" ")
    for t in tt:
      if len(t) != 0:
        words.add(t)

words = [(w, i) for i, w in enumerate(words)]
s = words[0]
for i in xrange(len(words)):
  if words[i][0] == "\n":
    s2 = i
words[s2] = (s[0], s2)
words[0] = ("\n", 0) 

f = open(path + "words.txt", 'w')
for w in words:
  f.write(w[0] + "\n")
cPickle.dump(words, open(path + "words.pkl", "wb" ))
words = dict(words)

for name in sets:
  fname = path + "ptb." + name
  data = []
  with open(fname + ".txt") as f: 
    text = f.readlines()
  for tt in text:
    tt = tt.split(" ")

    for t in tt:
      if len(t) != 0:
        data.append(words[t])

  data = np.array(data, dtype=np.int32)
  cPickle.dump(data, open(path + name + ".pkl", "wb" ))
