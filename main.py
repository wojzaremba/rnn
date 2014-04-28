#!/usr/bin/python
from model import Model
from layers.layer import MockSource, FCL, BiasL, TanhL, SigmL, ChrSource
from layers.cost import SoftmaxC
from layers.bundle import FCB, SoftmaxBC
import sys

# XXX: Why larger batch_size converge so much slower ?  
# XXX: Create tests !!! Itegrate provious testing env.

def mock(model):
  classes = 4
  model.n_epochs = 150
  model.set_source(MockSource, {'freq': 2, 'classes': classes, 'batch_size': 100, 'unroll': 5}) \
    .attach(FCL, {'out_len': 50, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(SigmL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': classes}) \
    .attach(SoftmaxC, {})
  return model

def pennchr(model):
  model.set_source(ChrSource, {'name': 'pennchr', 'batch_size': 5, 'unroll': 5}) \
    .attach(FCL, {'out_len': 50, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(SigmL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': 51}) \
    .attach(SoftmaxC, {})
  return model

def main():
  fun = 'mock'
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  model = Model(name=fun)
  model = eval(fun + '(model)')
  model.train()

if __name__ == '__main__':
  main()
