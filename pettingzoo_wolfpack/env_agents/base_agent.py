from abc import abstractmethod


class BaseAgent:

    @abstractmethod
    def step(self, observation) -> int:
        pass
