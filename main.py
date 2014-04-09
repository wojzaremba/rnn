import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

n = 50
nin = 5
nout = 5
n_t = 100

# input (where first dimension is time)
u = T.matrix()
# target (where first dimension is time)
t = T.matrix()
# initial hidden state of the RNN
h0 = T.vector()
# learning rate
lr = T.scalar()
# recurrent weights as a shared variable
W = theano.shared(np.random.uniform(size=(n, n), low=-.01, high=.01))
# input to hidden layer weights
W_in = theano.shared(np.random.uniform(size=(nin, n), low=-.01, high=.01))
# hidden to output layer weights
W_out = theano.shared(np.random.uniform(size=(n, nout), low=-.01, high=.01))


# recurrent function (using tanh activation function) and linear output
# activation function
def step(u_t, h_tm1, W, W_in, W_out):
  h_t = T.tanh(T.dot(u_t, W_in) + T.dot(h_tm1, W))
  y_t = T.dot(h_t, W_out)
  return h_t, y_t

# the hidden state `h` for the entire sequence, and the output for the
# entrie sequence `y` (first dimension is always time)
[h, y], _ = theano.scan(step,
                        sequences=u,
                        outputs_info=[h0, None],
                        non_sequences=[W, W_in, W_out])
# error between output and target
error = ((y - t) ** 2).sum()
# gradients on the weights using BPT
gW, gW_in, gW_out = T.grad(error, [W, W_in, W_out])
# training function, that computes the error and updates the weights using
# SGD.
updates = OrderedDict([(W, W - lr * gW),
                       (W_in, W_in - lr * gW_in),
                       (W_out, W_out - lr * gW_out)])
fn = theano.function([h0, u, t, lr], [error, y], updates=updates)

h0 = np.zeros((n,))
u0 = np.zeros((n_t, nin))
t = np.zeros((n_t, nin))
for i in xrange(n_t):
  u0[i, i % nin] = 1
  t[i, (i + 1) % nin] = 1

error, y = fn(h0, u0, t, 0.001)
print "y", y
