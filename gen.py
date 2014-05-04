#!/usr/bin/python
import sys

def main():
  fun = 'mock'
  text = "aabb"
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  if len(sys.argv) > 2:
    text = sys.argv[2]
  exec("from main import %s" % fun)
  model = eval(fun + '()')
  model.source.batch_size = 1
  model.init(-1, ask=False)
  model.gen(text)

if __name__ == '__main__':
  main()
