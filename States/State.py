import Constants


class State:
    def __init__(self, state_type, coords):
        self.state_type = state_type
        self.coords = coords

    def get_possible_actions(self, is_loaded=None, is_unloaded=None):
        return [Constants.left_action, Constants.right_action, Constants.up_action, Constants.down_action]
