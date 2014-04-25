import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
import pdb
from math import floor

n = 50
n_in = 5
n_out = 5
n_t = 20
u = T.ivector('u')
t = T.ivector('t')
h0 = T.vector('h0')
lr = T.scalar('lr')
W = theano.shared(np.random.uniform(size=(n, n), low=-.01, high=.01))
W_in = theano.shared(np.random.uniform(size=(n_in, n), low=-.01, high=.01))
W_out = theano.shared(np.random.uniform(size=(n, n_out), low=-.01, high=.01))

def step(u_t, t_t, h_tm1, W, W_in, W_out):
  h_t = T.tanh(W_in[u_t, :] + T.dot(h_tm1, W))
  y_t = T.dot(h_t, W_out)
  prob = T.nnet.softmax(y_t).T
  pred_t = T.argmax(prob, axis=0)[0]
  error_t = -T.log(prob[t_t]) 
  return h_t, pred_t, error_t

[_, preds, errors], _ = theano.scan(step,
                        sequences=[u, t],
                        outputs_info=[h0, None, None],
                        non_sequences=[W, W_in, W_out])

error = T.sum(errors)

gW, gW_in, gW_out = T.grad(error, [W, W_in, W_out])
updates = OrderedDict([(W, W - lr * gW),
                       (W_in, W_in - lr * gW_in),
                       (W_out, W_out - lr * gW_out)])
fn = theano.function([h0, u, t, lr], [preds, error], updates=updates)

for epoch in xrange(10000):
  h0 = np.zeros((n,))
  u0 = np.zeros(n_t, dtype=np.int32)
  t = np.zeros(n_t, dtype=np.int32)
  for i in xrange(n_t):
    u0[i] = floor((i + epoch) / 2) % n_in
    t[i] = floor((i + epoch + 1) / 2) % n_in

  preds, error = fn(h0, u0, t, 0.001)
  print "epoch ", epoch

  print "norm(W) = %.4f" % np.sum(W.get_value(borrow=False) ** 2)
  print "norm(W_in) = %.4f" % np.sum(W_in.get_value(borrow=False) ** 2)
  print "norm(W_out) = %.4f" % np.sum(W_out.get_value(borrow=False) ** 2)
  print "score %.3f" % np.mean(preds == t)
  print
