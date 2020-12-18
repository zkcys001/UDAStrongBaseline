from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss, SoftTripletLoss_uncer
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .multisoftmax import MultiSoftmaxLoss
__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'MultiSoftmaxLoss',
    'SoftTripletLoss_uncer',
]
