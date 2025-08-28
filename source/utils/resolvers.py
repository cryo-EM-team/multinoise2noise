from omegaconf import OmegaConf

def resolve_tuple(*args):
    return tuple(float(arg) for arg in args)

OmegaConf.register_new_resolver('as_tuple', resolve_tuple)

OmegaConf.register_new_resolver(
    "resolve_dtype",
    lambda s: getattr(torch, s.split('.')[-1])
)
