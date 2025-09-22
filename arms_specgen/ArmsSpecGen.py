from cores.TorchModelUtils.modeling import *

class ArmsSpecGen(ModelBase):
    def __init__(self):
        super(ArmsSpecGen, self).__init__(
            ignore_config_keys=[],
            **{k: v for k, v in locals().items() if k != 'self'}
        )