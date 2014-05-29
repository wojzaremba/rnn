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
  def __init__(self, name, lr=0.03, momentum=0.9, threshold=2., n_epochs=2):
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
    ret = ret + [h['layer'].get_hidden_output(name) for name, h in self.hiddens.iteritems()]
    return ret

  def build_model(self):
    print '\n... building the model with unroll=%d' \
      % self.source.unroll
    x = T.imatrix('x')
    y = T.imatrix('y')
    reset = T.scalar('reset')
    hiddens = [h['init'] for h in self.hiddens.values()]
    outputs_info = [None] * 3 + hiddens
    ret, updates = \
      theano.scan(self.step, sequences=[x, y], outputs_info=outputs_info)
    losses, probs, errors = ret[0], ret[1], ret[2]
    hids = ret[3:]
    loss = losses.sum()
    error = errors.sum() / T.cast((T.neq(y, 255).sum()), floatX)
    hidden_updates_train = []
    hidden_updates_test = []
    for h, ht in zip(self.hiddens.values(), hids):
      h_train = ifelse(T.eq(reset, 0), \
        ht[-1, :], T.ones_like(h['init']))
      h_test = ifelse(T.eq(reset, 0), \
        ht[-1, :], T.ones_like(h['init']))
      hidden_updates_train.append((h['init'], h_train))
      hidden_updates_test.append((h['init'], h_test))
    updates = self.source.get_updates(loss, self.sgd_params)
    updates += hidden_updates_train
    rets = [loss, probs[-1, :], error]
    mode = theano.Mode(linker='cvm')
    train_model = theano.function([x, y, reset, self.lr], rets, \
      updates=updates, mode=mode)
    test_model = theano.function([x, y, reset], rets, \
      updates=hidden_updates_test, mode=mode)
    return train_model, test_model

  def load(self, epoch=None, ask=True):
    if epoch == 0.:
      return 0.
    dname = conf.DUMP_DIR + self.name
    if not os.path.isdir(dname):
      return 0.
    if epoch is None:
      epochs = [float(f[len(self.name) + 1:]) for f in os.listdir(dname)]
      if len(epochs) == 0.:
        return 0.
      epoch = max(epochs)
    fname = "%s/%s_%.0f" % (dname, self.name, epoch)
    res = ''
    if epoch is None and ask:
      while res != "y" and res != "Y":
        res = raw_input("Resume %s (y), or reset (n) ? : " % fname)
        if res == "n" or res == "N":
          print "Starting training from beginning"
          return 0.
    print "Loading weights from %s ." % (fname)
    f = open(fname, 'rb')
    self.source.load(cPickle.load(f))
    print "Weights successfully loaded."
    for h in self.hiddens.values():
      h['init'].set_value(ONES(h['layer'].out_shape), borrow=False)
    return float(epoch) + 1.

  def save(self, epoch):
    if not os.path.isdir(conf.DUMP_DIR):
      os.makedirs(conf.DUMP_DIR)
    dname = conf.DUMP_DIR + self.name
    if not os.path.isdir(dname):
      os.makedirs(dname)
    fname = "%s/%s_%s" % (dname, self.name, str(epoch))
    f = open(fname, 'w')
    print "Saving weights %s" % (fname)
    cPickle.dump(self.source.dump(), f)

  def init(self, epoch=None, ask=True):
    self.source.rec_final_init()
    self.train_model, self.test_model = self.build_model()
    # XXX
    self.start_it = 0#self.load(epoch, ask)

  def gen(self, text_org=None, threshold=0):
    if self.start_it <= 0:
      print "Model not trained. Bye bye.\n"
      return
    if text_org is None:
      text_org = raw_input("Choose beginning of sequence:")
    print "Input sequence: %s" % text_org
    texts = []
    for k in xrange(10):
      np.random.seed(k)
      x = np.array([[ord(c) for c in text_org]], dtype=np.int32).transpose()
      y = np.zeros((len(x), 1), dtype=np.int32)
      for h in self.hiddens.values():
        h['init'].set_value(ONES((1, h['layer'].out_shape[1])), borrow=False)
      text = text_org
      rets = self.test_model(x, y, 0)
      for i in xrange(50):
        _, probs, _ = rets[0:3]
        p = [0]
        for i in xrange(probs.shape[1]):
          p.append(p[-1] + max(probs[0, i] - threshold, 0))
        assert abs(np.sum(probs) - 1) < 1e-2
        p = [v / p[-1] for v in p]
        u = np.random.uniform()
        idx = 0
        for i in xrange(len(p)):
          if u < p[i]:
            idx = i - 1
            break
        text += chr(idx)
        x = np.array([[idx]], dtype=np.int32)
        zero = np.array([[0]], dtype=np.int32)
        rets = self.test_model(x, zero, 0)
      print "Generated text : ", text
      texts.append(text)
    return texts

  def train(self):
    print '... training the model'
    start = time.time()
    last_save = start
    it = self.start_it
    lr = self.lr_val / self.source.batch_size
    perplexity = [float('Inf')]
    while True:
      data, epoch, last = self.source.get_train_data(it)
      if epoch >= self.n_epochs:
        break
      for i, (x, y) in enumerate(data):
        reset = i == len(data) - 1
        rets = self.train_model(x, y, reset, lr)
        loss, _, error = rets[0:3]
      if it % 10 == 2 or epoch == 0:
        elapsed = (time.time() - start) / 60
        data_iters = "epoch=%.0f, it=%d" % (epoch, it)
        scores = "loss=%f, error=%f, best validation perplexity = %f" \
          % (loss, error, min(perplexity))
        print "%s, %s, lr=%f, time elapsed=%.1f min." \
              % (data_iters, scores, lr, elapsed)
      if time.time() - last_save > 60 * 20:
        last_save = time.time()
        self.save(it)
      it += 1
      if last:
        perplexity.append(self.test(self.source.get_valid_data, False))
        if perplexity[-1] > min(perplexity):
          lr /= 2
        else:
          self.save(float('inf'))
        lp = len(perplexity)
        if (lp > 3 and min(perplexity[-3:]) > min(perplexity) * 1.005) or \
           (lp > 10 and min(perplexity[-10:]) > min(perplexity)):
          break
    self.start_it = it
    self.save(it)
    self.load(float('inf'), False) # Loading model with the best perplexity.
    print "Training finished !"
    print "Perplexities: %s" % str(perplexity)

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
        reset = i == len(data) - 1
        count += np.sum(y != 255)
        rets = self.test_model(x, y, reset)
        loss, _, error = rets[0:3]
        losses += loss
      if printout:
        print "it=%d, loss=%f, error=%f" % (it, loss, error)
    losses = losses / count
    perplexity = np.exp(losses)
    print "perplexity = %f\n" % perplexity
    return perplexity

