#!/usr/bin/python
from model import Model
from layers.layer import MockSource, FCL, BiasL, TanhL, TextSource, LSTML, RANDN, EYE
from layers.cost import SoftmaxC
import sys


def mock_lstm():
  classes = 4
  params = {'freq': 2, 'classes': classes, 'batch_size': 10, \
            'unroll': 6}
  model = Model(name="mock_lstm", n_epochs=15, lr=1., momentum=0., threshold=20.)
  model.set_source(MockSource, params) \
    .attach(FCL, {'out_len': 20}) \
    .attach(LSTML, {'out_len': 20, 'init': RANDN}) \
    .attach(FCL, {'out_len': 256}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model

def mock():
  classes = 4
  params = {'freq': 2, 'classes': classes, 'batch_size': 10, \
            'unroll': 5}
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

def pennlstm2():
  model = Model(name="pennlstm", n_epochs=10000, momentum=0., lr=1., threshold=100.)
  params = {'name': 'pennword', 'unroll': 20, 'out_len': 10000, 'batch_size': 50}
  model.set_source(TextSource, params) \
    .attach(FCL, {'out_len': 200}) \
    .attach(LSTML, {'out_len': 200, 'init': RANDN}) \
    .attach(FCL, {'out_len': 10000}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model

def pennlstmbs():
  model = Model(name="pennlstm", n_epochs=10000, momentum=0., lr=1., threshold=100.)
  params = {'name': 'pennword', 'unroll': 20, 'out_len': 10000, 'batch_size': 100}
  model.set_source(TextSource, params) \
    .attach(FCL, {'out_len': 200}) \
    .attach(LSTML, {'out_len': 200, 'init': RANDN}) \
    .attach(FCL, {'out_len': 10000}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model


def pennlstmbig():
  model = Model(name="pennlstm", n_epochs=10000, momentum=0., lr=1., threshold=100.)
  params = {'name': 'pennword', 'unroll': 50, 'out_len': 10000, 'batch_size': 100}
  model.set_source(TextSource, params) \
    .attach(FCL, {'out_len': 200}) \
    .attach(LSTML, {'out_len': 200, 'init': RANDN}) \
    .attach(FCL, {'out_len': 10000}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model


def pennword():
  model = Model(name="pennword", n_epochs=10000, momentum=0.5, lr=0.5)
  params = {'name': 'pennword', 'batch_size': 50, 'unroll': 20, 'out_len': 10000}
  model.set_source(TextSource, params) \
    .attach(FCL, {'out_len': 50, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(TanhL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': 10000}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model

def pennchr():
  model = Model(name="pennchr", n_epochs=10000, momentum=0.5, lr=0.5)
  params = {'name': 'pennchr', 'unroll': 20, 'out_len': 256}
  model.set_source(TextSource, params) \
    .attach(FCL, {'out_len': 600, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(TanhL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': 256}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  return model

def main():
  fun = 'mock_lstm'
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  model = eval(fun + '()')
  model.name = fun
  model.init()
  model.train()
  model.test()
  model.gen("aa")

if __name__ == '__main__':
  main()
