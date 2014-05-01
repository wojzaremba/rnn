import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from math import floor, ceil
import conf
import cPickle
import random

floatX = theano.config.floatX
RANDN = lambda s: 0.001 * np.array(np.random.random(s), dtype=floatX)
ZEROS = lambda s: np.zeros(s, dtype=floatX)
ONES = lambda s: np.ones(s, dtype=floatX)

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
    np.random.seed(1)
    W = theano.shared(value=init(size), name=name, borrow=False)
    dW = theano.shared(value=ZEROS(size), name=name, borrow=False)
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
    h = theano.shared(value=ZEROS(self.out_shape), name=name, borrow=False)
    self.model.hiddens[name]['init'] = h
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
    for l in self.succ:
      update = l.get_updates(cost, argv)
      updates += update
    for p, dp in zip(self.params, self.dparams):
      grad = T.grad(cost=cost, wrt=p)
      grad = momentum * dp + (1 - momentum) * grad
      grad *= ifelse(T.lt(l2(grad), threshold), 1., threshold / l2(grad))
      updates.append((dp, grad))
      updates.append((p, p - lr * grad))
    return updates

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
      params.append(p.get_value(borrow=False))
    for dp in self.dparams:
      dparams.append(dp.get_value(borrow=False))
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
      self.add_shared("W_%s" % name, (s[1], self.out_len))

  def fp(self, x, _):
    if x.type.dtype == 'int32':
      self.output = self.params[0][:, x].T
    else:
      self.output = T.dot(x, self.params[0].T)
    for i in xrange(len(self.hiddens)):
      name = self.hiddens[i]
      h = self.model.hiddens[name]['val']
      self.output += T.dot(h, self.params[i + 1])


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
  def __init__(self, model, unroll):
    Layer.__init__(self, None, model)
    self.unroll = unroll

  def get_train_data(self, it, rollover=True):
    fail

  def get_valid_data(self, it, rollover=True):
    return self.get_train_data(it, rollover)

  def get_test_data(self, it, rollover=True):
    return self.get_train_data(it, rollover)

  def fp(self, x, _):
    self.output = x

  def split(self, x):
    y = x[1:, :]
    x = x[0:-1, :]
    s = int(ceil(float(x.shape[0]) / float(self.unroll)))
    x = np.array_split(x, s)
    y = np.array_split(y, s)
    return zip(x, y)

class ChrSource(Source):
  def __init__(self, model, unroll, name):
    Source.__init__(self, model, unroll)
    self.name = name
    self.batch_size = None
    self.training = self.read_file("train.pkl")
    self.valid = self.read_file("valid.pkl")
    self.test = self.read_file("test.pkl")
    self.batch_size = self.training[0].shape[1]
    self.out_shape = (self.batch_size, 256)
    
  def read_file(self, filename):
    fname = conf.DATA_DIR + self.name + "/" + filename
    ret = cPickle.load(open(fname, "rb"))
    return ret

  def get_data(self, data, it, rollover=True):
    bs = self.batch_size
    s = len(data)
    epoch = int(floor(it/s))
    np.random.seed(epoch)
    it_perm = np.random.permutation(s)[it % s]
    x = data[it_perm]
    if rollover:
      return self.split(x), epoch
    else:
      last = (it%s+1)*bs >= data.shape[1]
      return self.split(x), last

  def get_train_data(self, it, rollover=True):
    return self.get_data(self.training, it, rollover)

  def get_valid_data(self, it, rollover=True):
    return self.get_data(self.valid, it, rollover)

  def get_test_data(self, it, rollover=True):
    return self.get_data(self.test, it, rollover)

class MockSource(Source):
  def __init__(self, model, batch_size, unroll, freq, classes):
    Source.__init__(self, model, unroll)
    self.batch_size = batch_size
    self.freq = freq
    self.classes = classes
    self.out_shape = (self.batch_size, 256)
    np.random.seed(1)

  def get_train_data(self, it, rollover=True):
    start = np.random.randint(0, 10)
    l = np.random.randint(0, 100) + 100 + 2 * self.unroll
    x = np.zeros(shape=(l, self.batch_size), dtype=np.int32)
    for b in xrange(self.batch_size):
      for i in xrange(l):
        x[i, b] = ord('a') + floor((i + b + start) / self.freq) % self.classes
    return self.split(x), floor(it * self.batch_size / 200)

  def get_test_data(self, it, rollover=True):
    data, epoch = self.get_train_data(it, rollover)
    if rollover:
      return data, epoch
    else:
      last = it > 2
      return data, last

def l2(x):
  return T.sqrt(T.sum(T.square(x)))
