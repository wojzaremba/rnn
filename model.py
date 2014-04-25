import numpy as np

import theano
import theano.tensor as T


class Model(object):
  def __init__(self, lr=0.01, n_epochs=50000):
    self.lr = lr
    self.n_epochs = n_epochs
    self.source = None
    self.hiddens = {}

  def set_source(self, source, params):
    params['model'] = self
    self.source = source(**params)
    return self.source

  def step(self, x_t, y_t, *hs):
    costs = self.source.get_costs(x_t, y_t)
    loss_t = costs[0].output
    pred_t = costs[0].pred()
    error_t = costs[0].acc(y_t)
    ret = [loss_t, pred_t, error_t]
    ret = [h['layer'].output for h in self.hiddens.values()] + ret
    return ret

  def build_model(self):
    print '\n... building the model'
    x = T.imatrix('x')
    y = T.imatrix('y')
    hiddens = [h['var'] for h in self.hiddens.values()]
    outputs_info = hiddens + [None] * 3
    [hids, losses, preds, errors], _ = theano.scan(self.step, sequences=[x, y], 
                            outputs_info=outputs_info)
    loss = T.sum(losses)
    error = T.mean(errors)
    updates = self.source.get_updates(self.lr, loss)
    train_model = theano.function(hiddens + [x, y], [hids[-1, :, :], loss, preds, error], updates=updates)
    return train_model

  def train(self):
    self.source.rec_final_init()
    train_model = self.build_model()
    print '... training the model'
    bs = self.source.batch_size
    hids = [np.zeros(h['layer'].out_shape) for h in self.hiddens.values()]
    for e in xrange(self.n_epochs):
      x, y = self.source.get_data(e)
      hids, loss, preds, error = train_model(*(hids + [x, y]))
      hids = [hids]
      print "x\n", x.transpose()
      print "y\n", y.transpose()
      print "preds\n", preds.transpose()
      print "e = %d, loss = %f, error = %f" % (e, loss, error)
      print 
