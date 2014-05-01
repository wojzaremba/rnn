import numpy as np
from math import exp, log
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import conf
import os
import cPickle
import time
from layers.layer import ONES

floatX = theano.config.floatX
class Model(object):
  def __init__(self, name, lr=1, momentum=0.5, threshold=15, n_epochs=2):
    print "_" * 100
    print "Creating model %s lr=%f, momentum=%f, n_epochs=%d" % \
      (name, lr, momentum, n_epochs)
    self.name = name
    self.lr = T.scalar('lr')
    self.lr_val = float(lr)
    self.sgd_params = (self.lr, momentum, threshold)
    self.n_epochs = n_epochs
    self.source = None
    self.train_model = None
    self.test_model = None
    self.start_it = 0
    self.hiddens = {}

  def set_source(self, source, params):
    params['model'] = self
    self.source = source(**params)
    return self.source

  def step(self, x_t, y_t, *hs):
    for i, k in enumerate(self.hiddens.keys()):
      self.hiddens[k]['val'] = hs[i]
    costs = self.source.get_costs(x_t, y_t)
    loss_t = costs[0].output
    prob_t = costs[0].prob
    error_t = T.cast(costs[0].error(y_t), floatX)
    ret = [loss_t, prob_t, error_t]
    ret = ret + [h['layer'].output for h in self.hiddens.values()] 
    return ret

  def build_model(self):
    print '\n... building the model'
    x = T.imatrix('x')
    y = T.imatrix('y')
    reset = T.scalar('reset')
    hiddens = []
    for h in self.hiddens.values():
      h_init = ifelse(T.eq(reset, 0), h['init'], T.ones_like(h['init']))
      hiddens.append(h_init)
    outputs_info = [None] * 3 + hiddens
    [losses, probs, errors, hids], updates = theano.scan(self.step, sequences=[x, y], 
                                                         outputs_info=outputs_info)
    loss = losses.sum()
    error = errors.sum() / T.cast((T.neq(y, 255).sum()), floatX)
    hidden_updates = [(h['init'], hids[-1]) for h in self.hiddens.values()]
    updates = self.source.get_updates(loss, self.sgd_params)
    updates += hidden_updates
    rets = [loss, probs[-1], error]
    train_model = theano.function([x, y, reset, self.lr], rets, updates=updates)
    test_model = theano.function([x, y, reset], rets, updates=hidden_updates)
    return train_model, test_model

  def load(self, ask=True):
    dname = conf.DUMP_DIR + self.name
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
    if not os.path.isdir(conf.DUMP_DIR):
      os.makedirs(conf.DUMP_DIR)
    dname = conf.DUMP_DIR + self.name
    if not os.path.isdir(dname):
      os.makedirs(dname)
    fname = "%s/%s_%d" % (dname, self.name, epoch)
    f = open(fname, 'w')
    print "Saving weights %s" % (fname)
    cPickle.dump(self.source.dump(), f)

  def init(self, ask=True):
    self.source.rec_final_init()
    self.train_model, self.test_model = self.build_model()
    self.start_it = self.load(ask)

  def gen(self, text_org=None):
    if self.start_it <= 0:
      print "Model not trained. Bye bye.\n"
      return
    if text_org is None:
      text_org = raw_input("Choose beginning of sequence:")
    print "Input sequence: %s" % text_org
    for k in xrange(5):
      np.random.seed(k)
      x = np.array([[ord(c) for c in text_org]], dtype=np.int32).transpose()
      y = np.zeros((len(x), 1), dtype=np.int32)
      text = text_org
      rets = self.test_model(x, y, 0)
      for i in xrange(100):
        loss, probs, error = rets[0:3]
        p = [0]
        for i in xrange(probs.shape[1]):
          p.append(p[-1] + probs[0, i])
        u = np.random.uniform()
        idx = 0
        for i in xrange(len(p)):
          if u < p[i]:
            idx = i - 1
            break
        text += chr(idx)
        x = np.array([[idx]], dtype=np.int32)
        zero = np.array([[0]], dtype=np.int32)
        rets = self.test_model(x, zero, 1)
      print "Generated text : ", text
   
  def train(self):
    print '... training the model'
    bs = self.source.batch_size
    start = time.time()
    last_save = start
    it = self.start_it
    perplexity = float('Inf')
    lr = self.lr_val / self.source.batch_size
    while True:  
      data, epoch, last = self.source.get_train_data(it)
      if epoch >= self.n_epochs:
        break
      for i, (x, y) in enumerate(data):
        reset = i == 0
        rets = self.train_model(x, y, reset, lr)
        loss, probs, error = rets[0:3]
      print "epoch=%d, it=%d, loss=%f, error=%f, lr=%f, since beginning=%.1f min." % (epoch, it, loss, error, lr, (time.time() - start) / 60)
      if time.time() - last_save > 60 * 10:
        last_save = time.time()
        self.save(it)
      it += 1
      if last:
        new_perplexity = self.test(self.source.get_valid_data, False)
        if new_perplexity > perplexity:
          lr /= 2
        perplexity = min(perplexity, new_perplexity)
    self.save(it - 1)
    print "Training finished !"

  def test(self, data_source=None, printout=True):
    if data_source == None:
      data_source = self.source.get_test_data
    if printout:
      print "\nTesting"
      print "_" * 100
    losses = 0
    count = 0
    last = False
    it = 0
    while not last:
      data, _, last = data_source(it)
      it += 1
      for i, (x, y) in enumerate(data):
        reset = i == 0
        count += np.sum(y != 255)
        rets = self.test_model(x, y, reset)
        loss, _, error = rets[0:3]
        losses += loss
      if printout:
        print "it=%d, loss=%f, error=%f" % (it, loss, error)
    losses = log(exp(1), 2) * losses / count
    perplexity = 2 ** losses
    print "perplexity = %f\n" % perplexity
    return perplexity

