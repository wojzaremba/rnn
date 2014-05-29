#!/usr/bin/python
import csv
import random
import cPickle

splits = [",", ".", ")", "(", "-", " ", "/", "#"]
top = 10000

def get_words(fname):
  with open(fname) as f:
    lines = f.readlines()
  words = {}
  for l in lines:
    for s in splits:
      l = l.replace(s, " ")
    l = l.lower()
    ww = l.split(" ")
    for w in ww:
      if len(w) >= 1:
        if not words.has_key(w):
          words[w] = 1
        else:
          words[w] += 1

  words = [(v, k) for k, v in words.iteritems()]
  words = sorted(words)
  words = words[-10000:]
  words = [b for a, b in words]
  return words

def get_data(fname):
  data = []
  active = True
  with open(fname) as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"): 
      if len(line) > 3:
        if len(line[1]) > 3 and len(line[4]) > 3:
          active = True
          data.append(([], line[1], line[4], line[2]))
        else:
          active = False
      else:
        if active:
          data[-1][0].append(line[2])
  return data

def translate(string, words):
  sequence = []
  string = string.lower()
  pos = 0
  end = pos
  while end < len(string):
    while end < len(string) and string[end] not in splits: 
      end += 1
    word = string[pos:end]
    if words.has_key(word):
      sequence.append(words[word])
    else:
      end += 1
      for i in xrange(pos, min(end, len(string))):
        sequence.append(ord(string[i]) + top)
    pos = end
  return sequence

def main():
  fname = "/Users/wojto/data/triplets/_people_person_children.tsv"
  #fname = "/Users/wojto/data/triplets/tmp.tsv"
  print "Getting words."
  words = get_words(fname)
  words = dict([(w, i) for i, w in enumerate(words)])
  print "Getting data."
  data = get_data(fname)

  print "Translating."
  all_data = []
  for single in data:
    for s in single[0]:
      text = s + "#" + single[1] + "," + single[2] + "," + single[3]
      t = translate(s, words)
      all_data.append(t)

  random.shuffle(all_data)
  idx = (9 * len(all_data)) / 10;
  print "Size of all data : %d" % len(all_data)
  print "Size of training data : %d" % idx
  train = all_data[:idx]
  test = all_data[idx:]
  f = open('/Users/wojto/data/triplets/train.pkl','w')
  cPickle.dump(train, f)
  f = open('/Users/wojto/data/triplets/test.pkl','w')
  cPickle.dump(test, f)

if __name__ == '__main__':
  main()
