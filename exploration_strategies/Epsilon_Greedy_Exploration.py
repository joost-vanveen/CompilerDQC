from exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy
import numpy as np
import random
import torch

class Epsilon_Greedy_Exploration(Base_Exploration_Strategy):
    """Implements an epsilon greedy exploration strategy"""
    def __init__(self, config):
        super().__init__(config)
        self.notified_that_exploration_turned_off = False
        if "exploration_cycle_episodes_length" in self.config.hyperparameters.keys():
            print("Using a cyclical exploration strategy")
            self.exploration_cycle_episodes_length = self.config.hyperparameters["exploration_cycle_episodes_length"]
            #print("######################**************************######################")
        else:
            self.exploration_cycle_episodes_length = None
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if "random_episodes_to_run" in self.config.hyperparameters.keys():
            self.random_episodes_to_run = self.config.hyperparameters["random_episodes_to_run"]
            print("Running {} random episodes".format(self.random_episodes_to_run))
        else:
            self.random_episodes_to_run = 0

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        episode_number = action_info["episode_number"]
        current_state = action_info["current_state"]
        if turn_off_exploration and not self.notified_that_exploration_turned_off:
            print(" ")
            print("Exploration has been turned OFF")
            print(" ")
            self.notified_that_exploration_turned_off = True
        epsilon = self.get_updated_epsilon_exploration(action_info)
        #epsilon = self.get_updated_epsilon_exploration_dynamic(action_info)
        #print(epsilon)
        
        if (random.random() > epsilon or turn_off_exploration) and (episode_number >= self.random_episodes_to_run):
            #print('argmax employed')
            return torch.argmax(action_values).item()
        #print('random section') 
        
        if (random.random() > epsilon or turn_off_exploration) and (episode_number >= self.random_episodes_to_run):
            return torch.argmax(action_values).item()

        # Pick a completely random action
        # TODO: change this such that it picks a valid action and not a completly random one
        random_values = list(range(action_values.shape[1]))
        for i, value in enumerate(current_state):
            if value == float('inf'):
                random_values.remove(i+1)
        
        return random.choice(random_values)
    
              
    def get_updated_epsilon_exploration_dynamic(self, action_info, epsilon=1.0):
        """Gets the probability that we just pick a random action. This probability decays the more episodes we have seen"""
        episode_number = action_info["episode_number"]
        epsilon_decay_denominator = self.config.hyperparameters["epsilon_decay_rate_denominator"]
        
        if episode_number % epsilon_decay_denominator == 0 and episode_number > 0:
            epsilon = epsilon / (episode_number // epsilon_decay_denominator)
        

        if self.exploration_cycle_episodes_length is None:
            epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
        else:
            epsilon = self.calculate_epsilon_with_cyclical_strategy(episode_number)
        return epsilon  

    def get_updated_epsilon_exploration(self, action_info, epsilon=1.0):
        """Gets the probability that we just pick a random action. This probability decays the more episodes we have seen"""
        episode_number = action_info["episode_number"]
        epsilon_decay_denominator = self.config.hyperparameters["epsilon_decay_rate_denominator"]

        if self.exploration_cycle_episodes_length is None:
            epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
        else:
            epsilon = self.calculate_epsilon_with_cyclical_strategy(episode_number)
        return epsilon

    def calculate_epsilon_with_cyclical_strategy(self, episode_number):
        """Calculates epsilon according to a cyclical strategy"""
        max_epsilon = 0.5
        min_epsilon = 0.001
        increment = (max_epsilon - min_epsilon) / float(self.exploration_cycle_episodes_length / 2)
        cycle = [ix for ix in range(int(self.exploration_cycle_episodes_length / 2))] + [ix for ix in range(
            int(self.exploration_cycle_episodes_length / 2), 0, -1)]
        cycle_ix = episode_number % self.exploration_cycle_episodes_length
        epsilon = max_epsilon - cycle[cycle_ix] * increment
        return epsilon

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        return reward_info["reward"]

    def reset(self):
        """Resets the noise process"""
        pass
