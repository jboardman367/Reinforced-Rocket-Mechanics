from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from torch.nn import Sequential, Linear
from torch.optim import Adam


class DualRocketAgent:
    def __init__(self):
        self.main_network = MainNetwork()
        self.aerial_network = AerialNetwork()

    @classmethod
    def load_from_file(cls, fpath: str):
        pass

    def save_to_file(self, fpath: str):
        pass


class MainNetwork(ActorCriticAgent):
    def __init__(self):
        split = (98,)
        actor = DiscretePolicy(Sequential(Linear(1, 256), Linear(256, 256), Linear(256, 256), Linear(256, 98)), shape=split)
        critic = Sequential(Linear(1, 256), Linear(256, 256), Linear(256, 256), Linear(256, 98))
        optimizer = Adam([
            {"params": self.actor.parameters(), "lr": 1e-5},
            {"params": self.critic.parameters(), "lr": 1e-5}
        ])

        super().__init__(actor, critic, optimizer)


class AerialNetwork(ActorCriticAgent):
    def __init__(self):
        split = (98,)
        actor = DiscretePolicy(Sequential(Linear(1, 256), Linear(256, 256), Linear(256, 256), Linear(256, 98)),
                               shape=split)
        critic = Sequential(Linear(1, 256), Linear(256, 256), Linear(256, 256), Linear(256, 98))
        optimizer = Adam([
            {"params": self.actor.parameters(), "lr": 1e-5},
            {"params": self.critic.parameters(), "lr": 1e-5}
        ])

        super().__init__(actor, critic, optimizer)


class KickoffNetwork(ActorCriticAgent):
    def __init__(self):
        pass

