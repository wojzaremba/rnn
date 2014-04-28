import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from math import floor
import config
import cPickle
import random

RANDN = lambda s: 0.001 * np.random.random(s)
ZEROS = np.zeros

class Layer(object):
  def __init__(self, prev_layer, model=None):
    self.output = None
    self.prev = []
    self.in_shape = None
    self.succ = []
    self.params = []
    self.dparams = []
    if prev_layer is not None:
      self.model = prev_layer.model
      self.in_shape = prev_layer.out_shape
      self.prev.append(prev_layer)
      prev_layer.succ.append(self)
    else:
      assert model is not None
      self.model = model

  def add_shared(self, name, size, init=RANDN):
    W = theano.shared(value=init(size), name=name, borrow=True)
    dW = theano.shared(value=ZEROS(size), name=name, borrow=True)
    self.params.append(W)
    self.dparams.append(dW)

  def rec_final_init(self):
    self.final_init()
    for l in self.succ:
      l.rec_final_init()

  def final_init(self):
    pass
 
  def get_prev(self):
    assert len(self.prev) == 1
    return self.prev[0]

  def add_hidden(self, name):
    if name not in self.model.hiddens:
      self.model.hiddens[name] = {}
    self.model.hiddens[name]['layer'] = self
    self.model.hiddens[name]['var'] = T.matrix(name)
    return self

  def attach(self, layer, params):
    if layer.__class__.__name__ == "function":
      print "Bundle %s" % layer.__name__
    params['prev_layer'] = self
    l = layer(**params)
    if layer.__class__.__name__ != "function":
      print "Layer %s, input shape %s" % (layer.__name__, str(l.in_shape))
    return l

  def get_updates(self, cost, argv):
    lr, momentum, threshold = argv
    updates = []
    grads = []
    norms = []
    for l in self.succ:
      update, grad, norm = l.get_updates(cost, argv)
      updates += update
      grads += grad
      norms += norm
    for p, dp in zip(self.params, self.dparams):
      grad = T.grad(cost=cost, wrt=p)
      dp_tmp = momentum * dp + (1 - momentum) * grad
      dp_tmp = ifelse(T.lt(l2(dp_tmp), threshold), dp_tmp, threshold * dp_tmp / l2(dp_tmp))
      updates.append((dp, dp_tmp))
      updates.append((p, p - lr * dp_tmp))
      grads.append(l2(dp_tmp))
      norms.append(l2(p))
    return updates, grads, norms

  def get_costs(self, x, y):
    # Due to circular dependency.
    from layers.cost import Cost
    if isinstance(self, Cost):
      costs = [self]
    else:
      costs = []
    for l in self.succ:
      l.fp(x, y)
      costs = costs + l.get_costs(l.output, y)
    unique = []
    for item in costs:
      if item not in unique:
        unique.append(item)
    return unique

  def fp(x, y):
    fail

  def dump(self):
    params = []
    dparams = []
    for p in self.params:
      params.append(p.get_value(borrow=True))
    for dp in self.dparams:
      dparams.append(dp.get_value(borrow=True))
    parent_params = []
    for l in self.succ:
      parent_params.append(l.dump())
    return (params, dparams, parent_params)

  def load(self, (params, dparams, parent_params)):
    for p_idx in xrange(len(self.params)):
      self.params[p_idx].set_value(params[p_idx])
    for dp_idx in xrange(len(self.dparams)):
      self.dparams[dp_idx].set_value(dparams[dp_idx])
    for l_idx in xrange(len(self.succ)):
      self.succ[l_idx].load(parent_params[l_idx])


class FCL(Layer):
  def __init__(self, out_len, hiddens=[], prev_layer=None):
    Layer.__init__(self, prev_layer)
    in_len = reduce(lambda x, y: x * y, list(self.in_shape)[1:])
    self.hiddens = hiddens
    self.add_shared("W", (out_len, in_len))
    self.out_len = out_len
    self.out_shape = (self.in_shape[0], out_len)

  def final_init(self):
    for i in xrange(len(self.hiddens)):
      name = self.hiddens[i]
      s = self.model.hiddens[name]['layer'].out_shape
      self.add_shared("W_%s" % name, (self.out_len, s[1]))

  def fp(self, x, _):
    if x.type.dtype == 'int32':
      self.output = self.params[0][:, x].T
    else:
      self.output = T.dot(x, self.params[0].T)
    for i in xrange(len(self.hiddens)):
      name = self.hiddens[i]
      h = self.model.hiddens[name]['prev_var']
      self.output += T.dot(h, self.params[i + 1].T)


class BiasL(Layer):
  def __init__(self, prev_layer=None):
    Layer.__init__(self, prev_layer)
    self.add_shared("b", (self.in_shape[1], ), ZEROS)
    self.out_shape = self.in_shape

  def fp(self, x, _):
    self.output = x + self.params[0]

class ActL(Layer):
  def __init__(self, f, prev_layer=None):
    Layer.__init__(self, prev_layer)
    self.f = f
    self.out_shape = self.in_shape

  def fp(self, x, _):
    self.output = self.f(x)

class ReluL(ActL):
  def __init__(self, prev_layer=None):
    relu = lambda x: T.maximum(x, 0)
    ActL.__init__(self, relu, prev_layer)

class TanhL(ActL):
  def __init__(self, prev_layer=None):
    ActL.__init__(self, T.tanh, prev_layer)

class SigmL(ActL):
  def __init__(self, prev_layer=None):
    sigmoid = lambda x: 1. / (1 + T.exp(-x))
    ActL.__init__(self, sigmoid, prev_layer)

class Source(Layer):
  def __init__(self, model, batch_size, unroll):
    Layer.__init__(self, None, model)
    self.batch_size = batch_size
    self.unroll = unroll

  def get_train_data(self, epoch):
    fail

  def get_valid_data(self, epoch):
    return self.get_train_data(epoch)

  def get_test_data(self, epoch):
    return self.get_train_data(epoch)

  def fp(self, x, _):
    self.output = x

  def split(self, x):
    y = x[1:-1, :]
    x = x[0:-2, :]
    s = x.shape[0] / self.unroll
    x = np.array_split(x, s)
    y = np.array_split(y, s)
    return zip(x, y)

class ChrSource(Source):
  def __init__(self, model, batch_size, unroll, name):
    Source.__init__(self, model, batch_size, unroll)
    self.name = name
    self.out_shape = (self.batch_size, 51)
    self.training = self.read_file("train.pkl")
    self.valid = self.read_file("valid.pkl")
    self.test = self.read_file("test.pkl")
    
  def read_file(self, filename):
    fname = config.DATA_DIR + self.name + "/" + filename
    ret = cPickle.load(open(fname, "rb"))
    return ret

  def get_data(self, data, epoch):
    bs = self.batch_size
    x = data[:, epoch*bs:(epoch+1)*bs]
    return self.split(x)

  def get_train_data(self, epoch):
    return self.get_data(self.training, epoch)

  def get_valid_data(self, epoch):
    return self.get_data(self.valid, epoch)

  def get_test_data(self, epoch):
    return self.get_data(self.test, epoch)

class MockSource(Source):
  def __init__(self, model, batch_size, unroll, freq, classes):
    Source.__init__(self, model, batch_size, unroll)
    self.freq = freq
    self.classes = classes
    self.out_shape = (self.batch_size, self.classes)

  def get_train_data(self, epoch):
    start = random.randint(0, 10)
    l = random.randint(0, 100) + 2 * self.unroll
    x = np.zeros(shape=(l, self.batch_size), dtype=np.int32)
    for b in xrange(self.batch_size):
      for i in xrange(l):
        x[i, b] = floor((i + b + start) / self.freq) % self.classes
    return self.split(x)

def l2(x):
  return T.sqrt(T.sum(T.square(x)))
