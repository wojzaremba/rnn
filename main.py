#!/usr/bin/python
from model import Model
from layers.layer import MockSource, FCL, BiasL, TanhL, SigmL, ChrSource
from layers.cost import SoftmaxC
from layers.bundle import FCB, SoftmaxBC
import sys

# XXX: Create tests !!! Itegrate provious testing env.

def penn_data():
   source = ChrSource
   params = {'name': 'pennchr', 'unroll': 10}
   return source, params

def mock_data():
  source = MockSource
  classes = 4
  params = {'freq': 2, 'classes': classes, 'batch_size': 10, 'unroll': 5}
  return source, params

def mock(source):
  model = Model(name="mock", n_epochs=1)
  model.set_source(source[0], source[1]) \
    .attach(FCL, {'out_len': 50, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(SigmL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': 256}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model

def pennchr(source, hid=600):
  model = Model(name="pennchr%d" % hid, n_epochs=10, momentum=0.)
  model.set_source(source[0], source[1]) \
    .attach(FCL, {'out_len': hid, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(SigmL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': 256}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model

def pennchr1000(source):
  return pennchr(source, 1000)

def main():
  options = {1: ('mock_data', 'mock'), 2:('penn_data', 'pennchr'), 3:('penn_data', 'pennchr1000')}
  option = 1
  if len(sys.argv) > 1:
    option = sys.argv[1]
  source_name, fun = options[option]
  source = eval(source_name + '()')
  model = eval(fun + '(source)')
  model.name = fun
  model.init()
  model.train()
  model.test()

if __name__ == '__main__':
  main()
