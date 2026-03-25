# Stub for unidepth - placeholder that does nothing since teacher uses MoGe+DepthPro for depth
class DepthHead:
    """Placeholder depth head. Teacher uses MoGe+DepthPro fusion instead."""
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.use_moge_depthpro = False

    def forward(self, *args, **kwargs):
        return {}

    def __call__(self, *args, **kwargs):
        return {}
