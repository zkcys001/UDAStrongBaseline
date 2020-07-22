from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .multisoftmax import MultiSoftmaxLoss
from .invariance import InvNet
__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'MultiSoftmaxLoss',
    'InvNet',
]
