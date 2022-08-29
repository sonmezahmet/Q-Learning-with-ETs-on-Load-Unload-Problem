from States.State import State
import Constants


class UpperSlope(State):
    def __init__(self, coords):
        super().__init__(state_type=Constants.upper_slope, coords=coords)

        # Initialize Q tables
        self.q0 = {Constants.right_action: 0, Constants.up_action: 0, Constants.down_action: 0}
        self.q1 = {Constants.right_action: 0, Constants.up_action: 0, Constants.down_action: 0}
        self.q2 = {Constants.right_action: 0, Constants.up_action: 0, Constants.down_action: 0}

        # Initialize e tables (Only for lambda Q learning)
        self.e0 = {Constants.right_action: 0, Constants.up_action: 0, Constants.down_action: 0}
        self.e1 = {Constants.right_action: 0, Constants.up_action: 0, Constants.down_action: 0}
        self.e2 = {Constants.right_action: 0, Constants.up_action: 0, Constants.down_action: 0}

    def get_possible_actions(self, is_loaded=None, is_unloaded=None):
        return [Constants.right_action, Constants.up_action, Constants.down_action]