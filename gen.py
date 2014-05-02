#!/usr/bin/python
from model import Model
from layers.layer import EmptySource
import sys
from main import *

def main():
  fun = 'pennchr600'
  text = "My_name_is"
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  if len(sys.argv) > 2:
    text = sys.argv[2]
  source = (EmptySource, {'batch_size': 1})   
  model = eval(fun + '(source)')

  model.init(ask=False)
  model.gen(text)
  

if __name__ == '__main__':
  main()
