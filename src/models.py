import gluoncv
from gluoncv2.model_provider import get_model as glcv2_get_model
import mxnet as mx
from mxnet.gluon import nn


def get_seresnext101_32x4d(units, load=None):
    net = gluoncv.model_zoo.get_model('se_resnext101_32x4d', pretrained=True)
    with net.name_scope():
        in_channels = 2048
        net.output = nn.Dense(units=units, in_units=in_channels)
    net.output.initialize(mx.init.Xavier())
    
    if load is not None:
        net.load_parameters(load)
        
    net.hybridize(static_alloc=True, static_shape=True)
    
    return net


def get_resnext50_32x4d(units, load=None):
    net = gluoncv.model_zoo.resnext.resnext50_32x4d(pretrained=True)
    with net.name_scope():
        in_channels = 2048
        net.output = nn.Dense(units=units, in_units=in_channels)
    net.output.initialize(mx.init.Xavier())
    
    if load is not None:
        net.load_parameters(load)
        
    net.hybridize(static_alloc=True, static_shape=True)
    
    return net


def get_pnasnet5large(units, load=None):
    net = glcv2_get_model('pnasnet5large', pretrained=True)
    with net.name_scope():
        in_channels = 4320
        net.output = nn.HybridSequential(prefix='')
        net.output.add(
            nn.Flatten(),
            nn.Dropout(rate=0.5),
            nn.Dense(units=units, in_units=in_channels)
        )
    net.output.initialize(mx.init.Xavier())
    
    if load is not None:
        net.load_parameters(load)
    
    net.hybridize(static_alloc=True, static_shape=True)
    
    return net