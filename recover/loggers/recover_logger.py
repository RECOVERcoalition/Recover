from ray.tune.logger import Logger


class RecoverLogger(Logger):
    """
    Base class for logging things with recover.  Inherits from ray's logging.
    Just allows us to also store the Data object along with all the bits
    ray logging natively stores.
    """

    def __init__(self, config, logdir, data, trial=None):
        self.data = data
        super().__init__(config, logdir, trial)
