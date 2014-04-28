#!/usr/bin/python
from model import Model
import sys
from main import *

def main():
  fun = 'mock'
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  model = Model(name=fun)
  model = eval(fun + '(model)')
  model.gen("aabb")
  

if __name__ == '__main__':
  main()
