from .McDecoder import *

model_zoo = {
    'McDecoder': McDecoder
}

def model_provider(name, **kwargs):

    model_ret = model_zoo[name](**kwargs)
    
    return model_ret
