from abc import abstractmethod


class DistnBase:

    def __init__(self, params):
        self.params = params
        return

    @abstractmethod
    def get_distn_type(self):
        pass

    @staticmethod
    @abstractmethod
    def np_ll():
        # define the log likelihood function via numpy
        pass

    @abstractmethod
    def tf_nll(self, y_true, y_pred):
        pass

    @abstractmethod
    def interpret_predict_output_ts(self, yhat):
        pass

    @abstractmethod
    def ppf(self, *args, **kwargs):
        pass
