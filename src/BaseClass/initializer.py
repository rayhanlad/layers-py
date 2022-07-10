from abc import ABC, abstractmethod


class Initializer(ABC):

    def __init__(self) -> None:
        pass
    
    def __call__(self, shape):
        pass