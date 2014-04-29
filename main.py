#!/usr/bin/python
from model import Model
from layers.layer import MockSource, FCL, BiasL, TanhL, SigmL, ChrSource
from layers.cost import SoftmaxC
from layers.bundle import FCB, SoftmaxBC
import sys

# XXX: Create tests !!! Itegrate provious testing env.
# XXX: Account in perplexity for loss before end of sentence.
# XXX: Get similar size text in every minibatch.

def mock(model):
  model.n_epochs = 3
  classes = 4
  model.set_source(MockSource, {'freq': 2, 'classes': classes, 'batch_size': 10, 'unroll': 5}) \
    .attach(FCL, {'out_len': 50, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(SigmL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': 255}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model

def pennchr(model):
  model.n_epochs = 1
  model.set_source(ChrSource, {'name': 'pennchr', 'unroll': 10}) \
    .attach(FCL, {'out_len': 200, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(SigmL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': 256}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model

def main():
  fun = 'mock'
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  model = Model(name=fun)
  model = eval(fun + '(model)')
  model.init()
  model.train()
  model.test()

if __name__ == '__main__':
  main()
