from recover.loggers.recover_logger import RecoverLogger
from abc import ABCMeta, abstractmethod
import numpy as np


class TBXWriter(metaclass=ABCMeta):
    """
    The TBXWriter is a class that may be implemented to add logging
    functionality to tensorboard with interplay on ray.  An example
    would be subclassing this class to implement writing the architecture
    graph, or to log a histogram of activations or gradients.

    Subclasses must implement write_result.  write_result returns nothing,
    and is passed in a dictionary representing the result state of a
    single step() call on the tune.Trainable.  The subclass should
    then make use of the _file_writer attribute and write what it would
    like to the _file_writer.

    Subclasses can also implement the methods _init() and close().

    _init() should be implemented if a subclass has initialization logic
    it needs to do.  Subclasses should NOT override __init__, as the writer
    objects are always initialized in the same way by the RecoverTBXLogger
    class.

    close() should be implemented if a writer has something it would only like
    to write when the trial has finished.  An example of this is writing all
    the hyperparameters of the trial once at the end of the trial (it'd be
    really inefficient to write all hyperparams to disk everytime (thanks
    GIL >:( )))

    """

    def __init__(self, config, data, tbx_file_writer, trial=None):
        self.config = config
        self.data = data
        self.trial = trial
        self._file_writer = tbx_file_writer
        self._init()

    def _init(self):
        pass

    @abstractmethod
    def write_result(self, result):
        raise NotImplementedError

    def close(self):
        pass


class TBXDictWriter(TBXWriter):
    """
    This logic is almost directly copy/pasted from ray's
    logic for writing results to tensorboard.  However,
    ray natively writes everything within one logger class.
    This doesn't give us the interop we would like, especially
    as it is best practice to write to disk with the same file
    writer.  So this class just encapsulates that logic from ray
    in a way that's easier for us to do interop.

    Note that hparams will be written only after a trial has terminated.
    This logger automatically flattens nested dicts to show on TensorBoard:

        {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}
    """

    def _init(self):
        self.last_result = None

    def write_result(self, result):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        tmp = result.copy()
        for k in ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]:
            if k in tmp:
                del tmp[k]  # not useful to log these

        flat_result = flatten_dict(tmp, delimiter="/")
        path = ["ray", "tune"]
        valid_result = {}

        for attr, value in flat_result.items():
            full_attr = "/".join(path + [attr])
            if type(value) in VALID_SUMMARY_TYPES and not np.isnan(value):
                valid_result[full_attr] = value
                self._file_writer.add_scalar(full_attr, value, global_step=step)
            elif (type(value) == list and len(value) > 0) or (
                type(value) == np.ndarray and value.size > 0
            ):
                valid_result[full_attr] = value
                try:
                    self._file_writer.add_histogram(full_attr, value, global_step=step)
                # In case TensorboardX still doesn't think it's a valid value
                # (e.g. `[[]]`), warn and move on.
                except (ValueError, TypeError):
                    if log_once("invalid_tbx_value"):
                        logger.warning(
                            "You are trying to log an invalid value ({}={}) "
                            "via {}!".format(full_attr, value, type(self).__name__)
                        )

        self.last_result = valid_result

    def close(self):
        """We want to log the hyperparams at the end if the trial had results :)"""
        if self.last_result:
            flat_result = flatten_dict(self.last_result, delimiter="/")
            scrubbed_result = {
                k: value
                for k, value in flat_result.items()
                if type(value) in VALID_SUMMARY_TYPES
            }
            self._try_log_hparams(scrubbed_result)

    def _try_log_hparams(self, result):
        # TBX currently errors if the hparams value is None.
        flat_params = flatten_dict(self.trial.evaluated_params)
        scrubbed_params = {
            k: v for k, v in flat_params.items() if isinstance(v, self.VALID_HPARAMS)
        }

        removed = {
            k: v
            for k, v in flat_params.items()
            if not isinstance(v, self.VALID_HPARAMS)
        }
        if removed:
            logger.info(
                "Removed the following hyperparameter values when "
                "logging to tensorboard: %s",
                str(removed),
            )

        from tensorboardX.summary import hparams

        try:
            experiment_tag, session_start_tag, session_end_tag = hparams(
                hparam_dict=scrubbed_params, metric_dict=result
            )
            self._file_writer.file_writer.add_summary(experiment_tag)
            self._file_writer.file_writer.add_summary(session_start_tag)
            self._file_writer.file_writer.add_summary(session_end_tag)
        except Exception:
            logger.exception(
                "TensorboardX failed to log hparams. "
                "This may be due to an unsupported type "
                "in the hyperparameter values."
            )


class RecoverTBXLogger(RecoverLogger):
    """TensorBoardX Logger.

    Basically just wraps a bunch of different writng logic and makes
    all the writers use the same file writer.  To use a new
    logger wtih this, be sure to add "tbx_writers" to the config and
    pass the writers you'd like to use as a list.

    """

    # NoneType is not supported on the last TBX release yet.
    VALID_HPARAMS = (str, bool, np.bool8, int, np.integer, float, list)

    def _init(self):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            logger.error("pip install 'ray[tune]' to see TensorBoard files.")
            raise
        self._file_writer = SummaryWriter(self.logdir, flush_secs=30)

        writer_classes = self.config.get("tbx_writers", [])
        self.writers = [
            cls(self.config, self.data, self._file_writer) for cls in writer_classes
        ]

    def on_result(self, result):
        """For compatibility with old versions of ray"""
        self.log_result(result)

    def log_result(self, result):
        for writer in self.writers:
            writer.write_result(result)

        self._file_writer.flush()

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()

    def close(self):
        if self._file_writer is not None:
            for writer in self.writers:
                writer.close()

            self._file_writer.close()
