class CustomError(Exception):
    """Base class for other exceptions"""

    pass


class HistoricalDataBufferError(CustomError):
    """Raised when there sufficient degree days dont accumulate because not enough historical data has been called"""

    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)


class PredictionNeededError(CustomError):
    """Raised when there sufficient degree days dont accumulate because a prediction is needed"""

    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)
