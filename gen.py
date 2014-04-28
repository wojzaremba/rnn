#!/usr/bin/python
from model import Model
import sys
from main import *

def main():
  fun = 'mock'
  text = "aabb"
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  if len(sys.argv) > 2:
    text = sys.argv[2]
  model = Model(name=fun)
  model = eval(fun + '(model)')
  model.init(ask=False)
  model.gen(text)
  

if __name__ == '__main__':
  main()
