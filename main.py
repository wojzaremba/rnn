#!/usr/bin/python
from model import Model
from layers.layer import MockSource, FCL, BiasL, TanhL, ChrSource
from layers.cost import SoftmaxC
import sys

def mock(backroll=0):
  classes = 4
  params = {'freq': 2, 'classes': classes, 'batch_size': 10, \
            'unroll': 5, 'backroll': backroll}
  model = Model(name="mock", n_epochs=5)
  model.set_source(MockSource, params) \
    .attach(FCL, {'out_len': 50, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(TanhL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': 256}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model

def pennchr(hid):
  model = Model(name="pennchr%d" % hid, n_epochs=10000, momentum=0.5, lr=0.5)
  params = {'name': 'pennchr', 'unroll': 20, 'backroll': 5}
  model.set_source(ChrSource, params) \
    .attach(FCL, {'out_len': hid, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(TanhL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': 256}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model

def pennchr1000():
  return pennchr(1000)

def pennchr600():
  return pennchr(600)

def pennchr800():
  return pennchr(800)

def main():
  fun = 'mock'
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  model = eval(fun + '()')
  model.name = fun
  model.init()
  model.train()
  model.test()

if __name__ == '__main__':
  main()
