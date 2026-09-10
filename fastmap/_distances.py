import abc


class InputError(Exception):
    """Raised when a distance metric receives unsupported input."""


class Distance(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def get_name():
        pass

    @abc.abstractmethod
    def calculate(self, x, y) -> float:
        pass
