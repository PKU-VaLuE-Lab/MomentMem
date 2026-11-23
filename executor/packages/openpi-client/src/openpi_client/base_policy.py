import abc
from typing import Dict


class BasePolicy(abc.ABC):
    # Policies that can consume observations without replanning should override
    # this capability flag and ``observe``. Callers use the flag to preserve the
    # behavior of ordinary stateless policies.
    supports_observe: bool = False

    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def observe(self, obs: Dict) -> None:
        """Consume an observation without producing a new action plan."""
        del obs

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
