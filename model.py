import numpy as np
from math import exp, log
import theano
import theano.tensor as T
import config
import os
import cPickle
import time

class Model(object):
  def __init__(self, name, lr=0.1, momentum=0.99, threshold=1, n_epochs=10000):
    self.name = name
    self.sgd_params = (lr, momentum, threshold)
    self.n_epochs = n_epochs
    self.source = None
    self.hiddens = {}

  def set_source(self, source, params):
    params['model'] = self
    self.source = source(**params)
    return self.source

  def step(self, x_t, y_t, *hs):
    for i, k in enumerate(self.hiddens.keys()):
      self.hiddens[k]['prev_var'] = hs[i]
    costs = self.source.get_costs(x_t, y_t)
    loss_t = costs[0].output
    prob_t = costs[0].prob
    error_t = costs[0].acc(y_t)
    ret = [loss_t, prob_t, error_t]
    ret = [h['layer'].output for h in self.hiddens.values()] + ret
    return ret

  def build_model(self):
    print '\n... building the model'
    x = T.imatrix('x')
    y = T.imatrix('y')
    hiddens = [h['var'] for h in self.hiddens.values()]
    outputs_info = hiddens + [None] * 3
    [hids, losses, probs, errors], _ = theano.scan(self.step, sequences=[x, y], 
                            outputs_info=outputs_info)
    loss = T.sum(losses)
    error = T.mean(errors)
    updates, grads, norms = self.source.get_updates(loss, self.sgd_params)
    rets = [hids[-1, :], loss, probs, error] + grads + norms
    train_model = theano.function(hiddens + [x, y], rets, updates=updates)
    test_model = theano.function(hiddens + [x, y], rets)
    return train_model, test_model

  def load(self, ask=True):
    dname = config.DUMP_DIR + self.name
    if not os.path.isdir(dname):
      return 0
    epochs = [int(f[len(self.name) + 1:]) for f in os.listdir(dname)]
    if len(epochs) == 0:
      return 0
    epoch = max(epochs)
    fname = "%s/%s_%d" % (dname, self.name, epoch)
    res = ''
    if ask:
      while res != "y" and res != "Y":
        res = raw_input("Resume %s (y), or start from scratch (n) ? : " % (fname))
        if res == "n" or res == "N":
          print "Starting training from beginning"
          return 0
    print "Loading weights from %s ." % (fname)
    f = open(fname, 'rb')
    self.source.load(cPickle.load(f))
    print "Weights successfully loaded."
    return epoch + 1

  def save(self, epoch):
    if not os.path.isdir(config.DUMP_DIR):
      os.makedirs(config.DUMP_DIR)
    dname = config.DUMP_DIR + self.name
    if not os.path.isdir(dname):
      os.makedirs(dname)
    fname = "%s/%s_%d" % (dname, self.name, epoch)
    f = open(fname, 'w')
    print "Saving weights %s" % (fname)
    cPickle.dump(self.source.dump(), f)

  def gen(self, text=None):
    self.source.rec_final_init()
    _, test_model = self.build_model()
    start_epoch = self.load(False)
    if start_epoch <= 0:
      print "Model not trained. Bye bye.\n"
      return
    if text is None:
      text = raw_input("Choose beginning of sequence:")
    text = text.lower()
    print "Input sequence: %s" % text
    x = np.array([[ord(c) - ord('a') for c in text]], dtype=np.int32).transpose()
    y = np.zeros((len(x), 1), dtype=np.int32)
    hids_zero = [np.zeros((1, h['layer'].out_shape[1])) for h in self.hiddens.values()]
    hids = hids_zero
    rets = test_model(*(hids + [x, y]))
    for i in xrange(100):
      hids, loss, probs, error = rets[0:4]
      hids = [hids]
      probs = probs[-1]
      p = [0]
      for i in xrange(probs.shape[1]):
        p.append(p[-1] + probs[0, i])
      u = np.random.uniform()
      idx = 0
      for i in xrange(len(p)):
        if u < p[i]:
          idx = i - 1
          break
      text += chr(idx + ord('a'))
      x = np.array([[idx]], dtype=np.int32)
      zero = np.array([[0]], dtype=np.int32)
      rets = test_model(*(hids + [x, zero]))
    print "Generated text : ", text
   
  def train(self):
    self.source.rec_final_init()
    train_model, test_model = self.build_model()
    start_epoch = self.load()
    print '... training the model'
    bs = self.source.batch_size
    hids_zero = [np.zeros(h['layer'].out_shape) for h in self.hiddens.values()]
    save_freq = 100
    start = time.time()
    for epoch in range(start_epoch, self.n_epochs):
      hids = hids_zero
      data = self.source.get_train_data(epoch)
      for x, y in data:
        rets = train_model(*(hids + [x, y]))
        hids, loss, probs, error = rets[0:4]
        hids = [hids]
      rets = rets[4:]
      grads = rets[0:len(rets) / 2]
      norms = rets[len(rets) / 2 :]
      #print "x\n", x.transpose()
      #print "y\n", y.transpose()
      #print "probs\n", probs.transpose()
      #print "grads\n", grads
      #print "norms\n", norms
      print "epoch = %d, loss = %f, error = %f, since beginning = %.1f min." % (epoch, loss, error, (time.time() - start) / 60)
      if epoch % save_freq == 0 and epoch > 0:
        self.save(epoch)
    self.save(self.n_epochs - 1)
    print "Training finished !"
    # Testing.
    print "\nTesting"
    print "_" * 100
    losses = 0
    count = 0
    for epoch in xrange(10):
      data = self.source.get_test_data(epoch)
      hids = hids_zero
      for x, y in data:
        count += x.shape[0]
        rets = test_model(*(hids + [x, y]))
        hids, loss, probs, error = rets[0:4]
        losses += loss * x.shape[0]
        hids = [hids]
      print "e = %d, loss = %f, error = %f" % (epoch, loss, error)
    losses = log(exp(1), 2) * losses / count

    print "perplexity = %f" % np.power(losses, 2)
