from __future__ import absolute_import

from .triplet import SoftTripletLoss_vallia, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .multisoftmax import MultiSoftmaxLoss
from .invariance import InvNet
__all__ = [
    'SoftTripletLoss_vallia',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'MultiSoftmaxLoss',
    'InvNet',
]
