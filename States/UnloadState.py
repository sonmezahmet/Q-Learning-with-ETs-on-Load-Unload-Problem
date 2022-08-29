from States.State import State
import Constants


class UnloadState(State):
    def __init__(self, coords):
        super().__init__(state_type=Constants.unload_state, coords=coords)

        # Initialize Q tables
        self.q0 = {Constants.left_action: 0, Constants.right_action: 0, Constants.up_action: 0,
                   Constants.down_action: 0}
        self.q1 = {Constants.unload_action: 0, Constants.left_action: 0, Constants.right_action: 0,
                   Constants.up_action: 0, Constants.down_action: 0}
        self.q2 = {Constants.left_action: 0, Constants.right_action: 0, Constants.up_action: 0,
                   Constants.down_action: 0}

        # Initialize e tables (Only for lambda Q learning)
        self.e0 = {Constants.left_action: 0, Constants.right_action: 0, Constants.up_action: 0,
                   Constants.down_action: 0}
        self.e1 = {Constants.unload_action: 0, Constants.left_action: 0, Constants.right_action: 0,
                   Constants.up_action: 0, Constants.down_action: 0}
        self.e2 = {Constants.left_action: 0, Constants.right_action: 0, Constants.up_action: 0,
                   Constants.down_action: 0}

    def get_possible_actions(self, is_loaded=None, is_unloaded=None):
        possible_actions = super().get_possible_actions()

        # If the agent is not unloaded
        if is_loaded is True and is_unloaded is False:
            possible_actions.append(Constants.unload_action)

        return possible_actions