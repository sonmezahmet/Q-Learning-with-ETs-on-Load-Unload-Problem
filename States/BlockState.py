from States.State import State
import Constants


class BlockState(State):
    def __init__(self, coords):
        super().__init__(state_type=Constants.block_state, coords=coords)

    def get_possible_actions(self, is_loaded=None, is_unloaded=None):
        return []