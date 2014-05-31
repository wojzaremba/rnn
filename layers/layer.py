import numpy as np
import os
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.nnet import sigmoid
from math import floor, ceil
import conf
import cPickle
import urllib
import os.path


floatX = theano.config.floatX
RANDN = lambda s: 0.05 * np.array(np.random.randn(s[0], s[1]), dtype=floatX)
EYE = lambda s: np.array(0.05 * np.random.randn(s[0], s[1]) + np.eye(s[0], s[1]), dtype=floatX) 
ZEROS = lambda s: np.zeros(s, dtype=floatX)
ONES = lambda s: np.ones(s, dtype=floatX)

def roll(data, start, end):
  start = start % len(data)
  end = end % len(data)
  if start < end:
    return data[start:end]
  else:
    return np.concatenate((data[start:], data[:end]))

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

  def get_hidden_output(self, _):
    return self.output

  def add_shared(self, name, size, init=RANDN):
    np.random.seed(1)
    W = theano.shared(value=init(size), name=name, borrow=False)
    dW = theano.shared(value=ZEROS(size), name=name, borrow=False)
    self.params.append(W)
    self.dparams.append(dW)
    return self.params[-1]

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
    h = theano.shared(value=ONES(self.out_shape), name=name, borrow=False)
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

  def get_grads(self, cost):
    grads = []
    for l in self.succ:
      grads += l.get_grads(cost)
    for p, dp in zip(self.params, self.dparams):
      grad = T.grad(cost=cost, wrt=p)
      grads.append((p, dp, grad))
    return grads

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

  def fp(self, x, y):
    raise NotImplementedError()

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

class LSTML(Layer):
  def __init__(self, out_len, init, prev_layer=None):
    Layer.__init__(self, prev_layer)
    in_len = reduce(lambda x, y: x * y, list(self.in_shape)[1:])
    self.out_len = out_len
    self.out_shape = (self.in_shape[0], out_len)
    self.hidden_id = np.random.randint(10000)

    self.add_hidden("h_%d" % self.hidden_id)
    self.add_hidden("c_%d" % self.hidden_id)
    self.Wxi = self.add_shared("Wxi", (in_len, out_len), init)
    self.Whi = self.add_shared("Whi", (out_len, out_len), init)
    self.Wci = self.add_shared("Wci", (out_len, out_len), init)
    self.Bi  = self.add_shared("Bi", (self.out_shape[1], ), ZEROS)

    self.Wxf = self.add_shared("Wxf", (in_len, out_len), init)
    self.Whf = self.add_shared("Whf", (out_len, out_len), init)
    self.Wcf = self.add_shared("Wcf", (out_len, out_len), init)
    self.Bf  = self.add_shared("Bf", (self.out_shape[1], ), ZEROS)

    self.Wxc = self.add_shared("Wxc", (in_len, out_len), init)
    self.Whc = self.add_shared("Whc", (out_len, out_len), init)
    self.Bc  = self.add_shared("Bc", (self.out_shape[1], ), ZEROS)

    self.Wxo = self.add_shared("Wxo", (in_len, out_len), init)
    self.Who = self.add_shared("Who", (out_len, out_len), init)
    self.Wco = self.add_shared("Wco", (out_len, out_len), init)
    self.Bo  = self.add_shared("Bo", (self.out_shape[1], ), ZEROS)
    self.ct = None


  def fp(self, x, _):
    relu = lambda x: T.max(x, 0)
    h = self.model.hiddens["h_%d" % self.hidden_id]['val']
    c = self.model.hiddens["c_%d" % self.hidden_id]['val']
    it = sigmoid(T.dot(x, self.Wxi) + T.dot(h, self.Whi) + T.dot(c, self.Wci) + self.Bi)
    ft = sigmoid(T.dot(x, self.Wxf) + T.dot(h, self.Whf) + T.dot(c, self.Wcf) + self.Bf)
    self.ct = ft * c + it * T.tanh(T.dot(x, self.Wxc) + T.dot(h, self.Whc) + self.Bc)
    ot = sigmoid(T.dot(x, self.Wxo) + T.dot(h, self.Who) + T.dot(self.ct, self.Wco) + self.Bo)
    self.output = ot * T.tanh(self.ct)

  def get_hidden_output(self, name):
    if name == "h_%d" % self.hidden_id:
      return self.output
    if name == "c_%d" % self.hidden_id:
      return self.ct;
    assert(0)

class FCL(Layer):
  def __init__(self, out_len, hiddens=None, prev_layer=None):
    Layer.__init__(self, prev_layer)
    in_len = reduce(lambda x, y: x * y, list(self.in_shape)[1:])
    if hiddens is None:
      hiddens = []
    self.hiddens = hiddens
    self.W = self.add_shared("W", (out_len, in_len))
    self.out_len = out_len
    self.out_shape = (self.in_shape[0], out_len)
    self.Wh = None

  def final_init(self):
    if len(self.hiddens) > 0:
      name = self.hiddens[0]
      s = self.model.hiddens[name]['layer'].out_shape
      self.Wh = self.add_shared("W_%s" % name, (s[1], self.out_len))

  def fp(self, x, _):
    if x.type.dtype == 'int32':
      self.output = self.W[:, x].T
    else:
      self.output = T.dot(x, self.W.T)
    if self.Wh is not None:
      name = self.hiddens[0]
      h = self.model.hiddens[name]['val']
      self.output += T.dot(h, self.Wh)


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

  def get_train_data(self, it):
    raise NotImplementedError()

  def get_valid_data(self, it):
    return self.get_train_data(it)

  def get_test_data(self, it):
    return self.get_train_data(it)

  def fp(self, x, _):
    self.output = x

  def split(self, x):
    s = int(ceil(float(x.shape[0]-1) / float(self.unroll)))
    xlist = []
    ylist = []
    for i in xrange(s):
      start = max(i*self.unroll, 0)
      end = min((i+1)*self.unroll, x.shape[0] - 1)
      xlist.append(x[start:end, :])
      ylist.append(x[(start+1):(end+1), :])
    reset = [0] * len(xlist)
    reset[-1] = 1
    return zip(xlist, ylist, reset)

class TextSource(Source):
  def __init__(self, model, unroll, batch_size, out_len, name):
    Source.__init__(self, model, unroll)
    self.name = name
    self.batch_size = None
    self.training = self.read_file("train.pkl")
    self.valid = self.read_file("valid.pkl")
    self.test = self.read_file("test.pkl")
    self.locs = np.random.randint(0, self.training.shape[0], batch_size)
    self.locs[0] = 0
    self.batch_size = batch_size
    self.out_shape = (self.batch_size, out_len)

  def read_file(self, filename):
    dname = conf.DATA_DIR + "/" + self.name
    fname = dname + "/" + filename
    f = open(fname, "rb")
    ret = cPickle.load(f)
    print "Openning %s. It has %d words." % (fname, len(ret))
    return ret

  def get_train_data(self, it):
    data = self.training
    x = np.zeros((self.unroll, self.batch_size), dtype=np.int32)
    y = np.zeros((self.unroll, self.batch_size), dtype=np.int32)
    for i in xrange(self.batch_size):
      start = (self.locs[i] + it * self.unroll) % len(data)
      end = (self.locs[i] + (it + 1) * self.unroll) % len(data)
      x[:, i] = roll(data, start, end)
      y[:, i] = roll(data, start + 1, end + 1)
    epoch = ((it * self.unroll) / len(data))
    return [(x, y, 0)], epoch, 0

  def get_one_pass(self, data, it):
    x = np.zeros((self.unroll, self.batch_size), dtype=np.int32)
    y = np.zeros((self.unroll, self.batch_size), dtype=np.int32)
    locs = range(len(data)) 
    locs = locs[::int(floor(len(data) / self.batch_size))]
    locs = locs[:-1]
    for i in xrange(self.batch_size):
      start = (locs[i] + it * self.unroll) % len(data)
      end = (locs[i] + (it + 1) * self.unroll) % len(data)
      x[:, i] = roll(data, start, end)
      y[:, i] = roll(data, start + 1, end + 1)
    epoch = ((it * self.unroll * self.batch_size) / len(data))
    last = (it + 1) * self.unroll > (len(data) / self.batch_size)
    return [(x, y, 0)], 0, last

  def get_valid_data(self, it):
    return self.get_one_pass(self.valid, it)

  def get_test_data(self, it):
    return self.get_one_pass(self.test, it)

class MockSource(Source):
  def __init__(self, model, batch_size, unroll, freq, classes):
    Source.__init__(self, model, unroll)
    self.batch_size = batch_size
    self.freq = freq
    self.classes = classes
    self.out_shape = (self.batch_size, 256)
    np.random.seed(1)

  def get_train_data(self, it):
    start = np.random.randint(0, 10)
    l = np.random.randint(0, 100) + 100 + 2 * self.unroll
    x = np.zeros(shape=(l, self.batch_size), dtype=np.int32)
    for b in xrange(self.batch_size):
      for i in xrange(l):
        x[i, b] = ord('a') + floor((i + b + start) / self.freq) % self.classes
        epoch = floor(it * self.batch_size / 200.)
        last = floor((it + 1) * self.batch_size / 200.) != epoch
    return self.split(x), epoch, last

  def get_valid_data(self, it):
    return self.get_test_data(it)

  def get_test_data(self, it):
    data, epoch, _ = self.get_train_data(it)
    return data, epoch, it % 5 == 4

def l2(x):
  return T.sqrt(T.sum(T.square(x)))

