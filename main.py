#!/usr/bin/python
from model import Model
from layers.layer import MockSource, FCL, BiasL, TanhL
from layers.cost import SoftmaxC
from layers.bundle import FCB, SoftmaxBC

def main():
  model = Model()
  classes = 5
  model.set_source(MockSource, {'classes': classes}) \
    .attach(FCL, {'out_len': 50, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(TanhL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': classes}) \
    .attach(BiasL, {}) \
    .attach(SoftmaxC, {})
  model.train()


if __name__ == '__main__':
  main()
