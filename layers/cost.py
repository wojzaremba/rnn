import theano.tensor as T
from layers.layer import Layer
from theano.ifelse import ifelse

class Cost(Layer):
  def __init__(self, prev_layer):
    Layer.__init__(self, prev_layer)
    self.out_shape = (1,)
    self.output = None
    self.prob = None

  def error(self, y):
    pred =  T.argmax(self.prob, axis=1)
    return T.sum((T.neq(pred, y)) * T.neq(y, 255))

class SoftmaxC(Cost):
  def __init__(self, prev_layer=None):
    Cost.__init__(self, prev_layer)
    self.params = []

  def fp(self, x, y):
    self.prob = T.nnet.softmax(x)
    output = -T.mean(T.log(self.prob)[T.arange(y.shape[0]), y])
    self.output = T.sum(T.neq(y, 255) * output)

