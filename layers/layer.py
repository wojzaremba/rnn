import numpy as np
import theano
import theano.tensor as T
from math import floor

class Layer(object):
  def __init__(self, prev_layer, model=None):
    self.output = None
    self.prev = []
    self.in_shape = None
    self.succ = []
    if prev_layer is not None:
      self.model = prev_layer.model
      self.in_shape = prev_layer.out_shape
      self.prev.append(prev_layer)
      prev_layer.succ.append(self)
    else:
      assert model is not None
      self.model = model
 
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

  def get_updates(self, lr, cost):
    updates = []
    for l in self.succ:
      updates += l.get_updates(lr, cost)
    for p in self.params:
      g = T.grad(cost=cost, wrt=p)
      updates.append((p, p - lr * g))
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

class FCL(Layer):
  def __init__(self, out_len, hiddens=[], prev_layer=None):
    Layer.__init__(self, prev_layer)
    in_len = reduce(lambda x, y: x * y, list(self.in_shape)[1:])
    self.hiddens = hiddens
    
    self.params = []

    W = theano.shared(value=0.1 * np.random.randn(out_len, in_len),
                             name='W', borrow=True)
    self.params.append(W)
    self.out_len = out_len
    self.out_shape = (self.in_shape[0], out_len)
    self.further_init()


  def further_init(self):
    for i in xrange(len(self.hiddens)):
      W = theano.shared(value=0.01 * np.random.randn(self.out_len, 50),
                             name='W_%s' % self.hiddens[i], borrow=True)
      self.params.append(W)

  def fp(self, x, _):
    if x.type.dtype == 'int32':
      self.output = self.params[0][:, x].T
    else:
      self.output = T.dot(x, self.params[0].T)
    for i in xrange(len(self.hiddens)):
      name = self.hiddens[i]
      h = self.model.hiddens[name]['var']
      self.output += T.dot(h, self.params[i + 1].T)


class BiasL(Layer):
  def __init__(self, prev_layer=None):
    Layer.__init__(self, prev_layer)
    self.b = theano.shared(value=np.zeros((self.in_shape[1],),
                           dtype=theano.config.floatX),
                           name='b', borrow=True)
    self.out_shape = self.in_shape
    self.params = [self.b]

  def fp(self, x, _):
    self.output = x + self.b


class ActL(Layer):
  def __init__(self, f, prev_layer=None):
    Layer.__init__(self, prev_layer)
    self.f = f
    self.out_shape = self.in_shape
    self.params = []

  def fp(self, x, _):
    self.output = self.f(x)


class ReluL(ActL):
  def __init__(self, prev_layer=None):
    relu = lambda x: T.maximum(x, 0)
    ActL.__init__(self, relu, prev_layer)


class TanhL(ActL):
  def __init__(self, prev_layer=None):
    ActL.__init__(self, T.tanh, prev_layer)

class MockSource(Layer):
  def __init__(self, classes, model):
    Layer.__init__(self, None, model)
    self.batch_size = 100
    self.n_t = 20
    self.classes = classes
    self.out_shape = (self.batch_size, self.classes)
    self.params = []

  def get_data(self, epoch):
    x = np.zeros(shape=(self.n_t, self.batch_size), dtype=np.int32)
    y = np.zeros(shape=(self.n_t, self.batch_size), dtype=np.int32)
    for b in xrange(self.batch_size):
      for i in xrange(self.n_t):
        x[i, b] = floor((i + b + epoch) / 2) % self.classes
        y[i, b] = floor((i + 1 + b + epoch) / 2) % self.classes
    return x, y

  def fp(self, x, _):
    self.output = x

