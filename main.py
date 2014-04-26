#!/usr/bin/python
from model import Model
from layers.layer import MockSource, FCL, BiasL, TanhL, SigmL
from layers.cost import SoftmaxC
from layers.bundle import FCB, SoftmaxBC

def main():
  model = Model()
  classes = 4
  model.set_source(MockSource, {'freq': 3, 'classes': classes, 'batch_size': 1, 'n_t': 16}) \
    .attach(FCL, {'out_len': 50, 'hiddens' : ['qqq']}) \
    .attach(BiasL, {}) \
    .attach(SigmL, {}) \
    .add_hidden('qqq') \
    .attach(FCL, {'out_len': classes}) \
    .attach(SoftmaxC, {})
  model.train()


if __name__ == '__main__':
  main()
