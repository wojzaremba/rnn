from layers.layer import FCL, BiasL, ReluL
from layers.cost import SoftmaxC

def FCB(out_len, prev_layer=None):
  return prev_layer.attach(FCL, {'out_len': out_len})\
  .attach(BiasL, {})\
  .attach(ReluL, {})

def ConvB(filter_shape, subsample=(1, 1),
          border_mode='full', prev_layer=None):
  return prev_layer.attach(ConvL, {'filter_shape':filter_shape,
                            'subsample':subsample,
                            'border_mode':border_mode})\
  .attach(BiasL, {})\
  .attach(ReluL, {})

def SoftmaxBC(out_len, prev_layer=None):
  return prev_layer.attach(FCL, {'out_len': out_len})\
  .attach(BiasL, {})\
  .attach(SoftmaxC, {})
