from States.State import State
from States.UpperSlope import UpperSlope
from States.LowerSlope import LowerSlope
import Constants


class SlopeState(State):
    def __init__(self, coords):
        super().__init__(state_type=Constants.slope_state, coords=coords)

        # In slope states there are two different sub states which are the upper slope and lower slope
        # If the agent in upper slope, it can take only 3 possible moves which are right, up, and down actions
        # If the agent in lower slope, it can take only 3 possible moves which are left, up, and down actions
        self.upper_slope = UpperSlope(coords=coords)
        self.lower_slope = LowerSlope(coords=coords)
