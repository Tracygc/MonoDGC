from lib.models.monodgp import build_monodgp
from lib.models.monodgc import build_monodgc

def build_model(cfg):
    return build_monodgp(cfg)

def build_model7(cfg):
    return build_monodgc(cfg)