from States.State import State
import Constants


class LoadState(State):
    def __init__(self, coords):
        super().__init__(state_type=Constants.load_state, coords=coords)

        # Initialize Q tables
        self.q0 = {Constants.load_action: 0, Constants.left_action: 0, Constants.right_action: 0,
                   Constants.up_action: 0, Constants.down_action: 0}
        self.q1 = {Constants.left_action: 0, Constants.right_action: 0, Constants.up_action: 0,
                   Constants.down_action: 0}
        self.q2 = {Constants.left_action: 0, Constants.right_action: 0, Constants.up_action: 0,
                   Constants.down_action: 0}

        # Initialize e tables (Only for lambda Q learning)
        self.e0 = {Constants.load_action: 0, Constants.left_action: 0, Constants.right_action: 0,
                   Constants.up_action: 0, Constants.down_action: 0}
        self.e1 = {Constants.left_action: 0, Constants.right_action: 0, Constants.up_action: 0,
                   Constants.down_action: 0}
        self.e2 = {Constants.left_action: 0, Constants.right_action: 0, Constants.up_action: 0,
                   Constants.down_action: 0}

    def get_possible_actions(self, is_loaded=None, is_unloaded=None):
        possible_actions = super().get_possible_actions()

        # If the agent is not loaded
        if is_loaded is False:
            possible_actions.append(Constants.load_action)

        return possible_actions
