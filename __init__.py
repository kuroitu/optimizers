from ._interface import get_opt
from ._base import BaseOpt
from ._sgd import SGD
from ._msgd import MSGD
from ._nag import NAG
from ._adagrad import AdaGrad
from ._rmsprop import RMSprop
from ._adadelta import AdaDelta
from ._adam import Adam
from ._rmsprop_graves import RMSpropGraves
from ._smorms3 import SMORMS3
from ._adamax import AdaMax
from ._nadam import Nadam
from ._eve import Eve
from ._santa_e import SantaE
from ._santa_sss import SantaSSS
from ._amsgrad import AMSGrad
from ._adabound import AdaBound
from ._amsbound import AMSBound
from ._adabelief import AdaBelief
from ._sam import SAM


opt_dict = {"sgd": SGD,
            "msgd": MSGD,
            "nag": NAG,
            "adagrad": AdaGrad,
            "rmsprop": RMSprop,
            "adadelta": AdaDelta,
            "adam": Adam,
            "rmspropgraves": RMSpropGraves,
            "smorms3": SMORMS3,
            "adamax": AdaMax,
            "nadam": Nadam,
            "eve": Eve,
            "santae": SantaE,
            "santasss": SantaSSS,
            "amsgrad": AMSGrad,
            "adabound": AdaBound,
            "amsbound": AMSBound,
            "adabelief": AdaBelief,
            "sam": SAM}
