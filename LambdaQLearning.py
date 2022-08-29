import random
import Constants
import matplotlib.pyplot as plt


class LambdaQLearning:
    def __init__(self, environment, agent, learning_rate, discount_rate, epsilon, lambda_value):
        self.environment = environment
        self.agent = agent
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.lambda_value = lambda_value

    def train(self, num_of_episodes):
        train_dict = dict()
        for episode in range(num_of_episodes):
            # Initialize agent
            self.agent.current_state = self.environment.env[0][0]
            self.agent.is_loaded = False
            self.agent.is_unloaded = False

            # Initialize s, a
            action = self.__select_action(state=self.agent.current_state)
            current_state = self.agent.current_state

            # Reset all e tables
            self.__reset_e_tables()

            is_episode_finished = False
            step_count = 0
            while not is_episode_finished:
                # Take action, observe reward, and next state
                next_state, reward, is_episode_finished = self.__take_action(state=current_state, action=action)
                # Choose next_action from next_state using e-greedy policy
                next_action = self.__select_action(state=next_state)
                # a_asterisks = argmaxbQ(s’,b) (if a’ ties for the max, then a*=a’)
                a_asterisks = self.__select_action(state=next_state, greedy_selection=True)
                # Compute TD error
                if self.agent.is_loaded is False and self.agent.is_unloaded is False:
                    q_table = current_state.q0
                    e_table = current_state.e0
                    q_table_next = None
                    if action == Constants.load_action:
                        q_table_next = current_state.q1
                    else:
                        q_table_next = next_state.q0
                elif self.agent.is_loaded is True and self.agent.is_unloaded is False:
                    q_table = current_state.q1
                    e_table = current_state.e1
                    q_table_next = None
                    if action == Constants.unload_action:
                        q_table_next = current_state.q2
                    else:
                        q_table_next = next_state.q1
                else:
                    q_table = current_state.q2
                    e_table = current_state.e2
                    q_table_next = next_state.q2
                td_error = reward + self.discount_rate * max(q_table_next.values()) - q_table[action]
                #e_table[action] = 1
                for key in e_table:
                    if key == action:
                        e_table[key] = 1
                    else:
                        e_table[key] = 0
                # For all s,a:
                for i in range(self.environment.n):
                    for j in range(self.environment.n):
                        state = self.environment.env[i][j]

                        # If state is slope state, update both slopes (upper and lower slopes)
                        if state.state_type == Constants.slope_state:
                            # Update q0 table
                            for key in state.upper_slope.q0:
                                state.upper_slope.q0[key] = state.upper_slope.q0[key] + self.learning_rate * td_error\
                                                            * state.upper_slope.e0[key]

                                if state.upper_slope.q0 == current_state.q0:
                                    continue
                                else:
                                    # Update e table
                                    state.upper_slope.e0[key] = self.discount_rate * self.lambda_value * \
                                                                state.upper_slope.e0[key]

                                '''
                                if next_action == a_asterisks:
                                    state.upper_slope.e0[key] = self.discount_rate * self.lambda_value * \
                                                                state.upper_slope.e0[key]
                                else:
                                    state.upper_slope.e0[key] = 0
                                '''

                            for key in state.lower_slope.q0:
                                state.lower_slope.q0[key] = state.lower_slope.q0[key] + self.learning_rate * td_error\
                                                            * state.lower_slope.e0[key]

                                if state.lower_slope.q0 == current_state.q0:
                                    continue
                                else:
                                    # Update e table
                                    state.lower_slope.e0[key] = self.discount_rate * self.lambda_value * \
                                                                state.lower_slope.e0[key]

                                '''
                                if next_action == a_asterisks:
                                    state.lower_slope.e0[key] = self.discount_rate * self.lambda_value * \
                                                                state.lower_slope.e0[key]
                                else:
                                    state.lower_slope.e0[key] = 0
                                '''

                            # Update q1 table
                            for key in state.upper_slope.q1:
                                state.upper_slope.q1[key] = state.upper_slope.q1[key] + self.learning_rate * td_error \
                                                            * state.upper_slope.e1[key]

                                if state.upper_slope.q1 == current_state.q1:
                                    continue
                                else:
                                    # Update e table
                                    state.upper_slope.e1[key] = self.discount_rate * self.lambda_value * \
                                                                state.upper_slope.e1[key]

                                '''
                                if next_action == a_asterisks:
                                    state.upper_slope.e1[key] = self.discount_rate * self.lambda_value * \
                                                                state.upper_slope.e1[key]
                                else:
                                    state.upper_slope.e1[key] = 0
                                '''

                            for key in state.lower_slope.q1:
                                state.lower_slope.q1[key] = state.lower_slope.q1[key] + self.learning_rate * td_error \
                                                            * state.lower_slope.e1[key]

                                if state.lower_slope.q1 == current_state.q1:
                                    continue
                                else:
                                    # Update e table
                                    state.lower_slope.e1[key] = self.discount_rate * self.lambda_value * \
                                                                state.lower_slope.e1[key]

                                '''
                                if next_action == a_asterisks:
                                    state.lower_slope.e1[key] = self.discount_rate * self.lambda_value * \
                                                                state.lower_slope.e1[key]
                                else:
                                    state.lower_slope.e1[key] = 0
                                '''

                            # Update q2 table
                            for key in state.upper_slope.q2:
                                state.upper_slope.q2[key] = state.upper_slope.q2[key] + self.learning_rate * td_error \
                                                            * state.upper_slope.e2[key]

                                if state.upper_slope.q2 == current_state.q2:
                                    continue
                                else:
                                    # Update e table
                                    state.upper_slope.e2[key] = self.discount_rate * self.lambda_value * \
                                                                state.upper_slope.e2[key]

                                '''
                                if next_action == a_asterisks:
                                    state.upper_slope.e2[key] = self.discount_rate * self.lambda_value * \
                                                                state.upper_slope.e2[key]
                                else:
                                    state.upper_slope.e2[key] = 0
                                '''

                            for key in state.lower_slope.q2:
                                state.lower_slope.q2[key] = state.lower_slope.q2[key] + self.learning_rate * td_error \
                                                            * state.lower_slope.e2[key]

                                if state.lower_slope.q2 == current_state.q2:
                                    continue
                                else:
                                    # Update e table
                                    state.lower_slope.e2[key] = self.discount_rate * self.lambda_value * \
                                                                state.lower_slope.e2[key]

                                '''
                                if next_action == a_asterisks:
                                    state.lower_slope.e2[key] = self.discount_rate * self.lambda_value * \
                                                                state.lower_slope.e2[key]
                                else:
                                    state.lower_slope.e2[key] = 0
                                '''

                        else:
                            if state.state_type != Constants.block_state:
                                # Update q0 table
                                for key in state.q0:
                                    state.q0[key] = state.q0[key] + self.learning_rate * td_error * state.e0[key]

                                    if state.q0 == current_state.q0:
                                        continue
                                    else:
                                        # Update e table
                                        state.e0[key] = self.discount_rate * self.lambda_value * \
                                                                    state.e0[key]

                                    '''
                                    if next_action == a_asterisks:
                                        state.e0[key] = self.discount_rate * self.lambda_value * \
                                                                    state.e0[key]
                                    else:
                                        state.e0[key] = 0
                                    '''

                                # Update q1 table
                                for key in state.q1:
                                    state.q1[key] = state.q1[key] + self.learning_rate * td_error * state.e1[key]

                                    if state.q1 == current_state.q1:
                                        continue
                                    else:
                                        # Update e table
                                        state.e1[key] = self.discount_rate * self.lambda_value * \
                                                                    state.e1[key]

                                    '''
                                    if next_action == a_asterisks:
                                        state.e1[key] = self.discount_rate * self.lambda_value * \
                                                                    state.e1[key]
                                    else:
                                        state.e1[key] = 0
                                    '''

                                # Update q2 table
                                for key in state.q2:
                                    state.q2[key] = state.q2[key] + self.learning_rate * td_error * state.e2[key]

                                    if state.q2 == current_state.q2:
                                        continue
                                    else:
                                        # Update e table
                                        state.e2[key] = self.discount_rate * self.lambda_value * \
                                                                    state.e2[key]

                                    '''
                                    if next_action == a_asterisks:
                                        state.e2[key] = self.discount_rate * self.lambda_value * \
                                                                    state.e2[key]
                                    else:
                                        state.e2[key] = 0
                                    '''
                if action == Constants.load_action:
                    self.agent.is_loaded = True
                    if next_action == Constants.load_action:
                        # Select new next action
                        next_action = self.__select_action(state=next_state)
                elif action == Constants.unload_action:
                    self.agent.is_unloaded = True
                    if next_action == Constants.unload_action:
                        # Select new next action
                        next_action = self.__select_action(state=next_state)
                # s ← s’; a ← a’
                self.agent.previous_state = self.agent.current_state
                self.agent.current_state = next_state
                current_state = next_state
                action = next_action

                step_count += 1
            train_dict[episode] = step_count
            print(f'Episode {episode} is finished in {step_count} steps.')
            print('-' * 100)

        # Plot train results
        episodes = list(train_dict.keys())
        steps = list(train_dict.values())
        plt.plot(episodes, steps)
        plt.title('Lambda Q-Learning Training')
        plt.xlabel('Episode')
        plt.ylabel('Step Count')
        plt.show()

    def test(self):
        # Initialize the agent
        self.agent.current_state = self.environment.env[0][0]
        self.agent.previous_state = None
        self.agent.is_loaded = False
        self.agent.is_unloaded = False
        self.agent.total_reward = 0

        is_episode_finished = False
        step_count = 0
        while not is_episode_finished:
            # Select an action
            action = self.__select_action(state=self.agent.current_state, greedy_selection=True)

            # Take action; observe reward and next state
            next_state, reward, is_episode_finished = self.__take_action(state=self.agent.current_state, action=action)

            print(f'Step: {step_count}, Current State: {self.agent.current_state.coords}, Action: {action}, '
                  f'Next State: {next_state.coords}, Reward: {reward}')

            # Move agent to next state
            self.agent.total_reward += reward
            if action == Constants.load_action:
                self.agent.is_loaded = True
            elif action == Constants.unload_action:
                self.agent.is_unloaded = True
            self.agent.previous_state = self.agent.current_state
            self.agent.current_state = next_state

            step_count += 1

        print(f'Test is finished in {step_count} steps. Total Reward: {self.agent.total_reward}')

    def __reset_e_tables(self):
        for i in range(self.environment.n):
            for j in range(self.environment.n):
                state = self.environment.env[i][j]

                if state.state_type == Constants.slope_state:
                    # Reset both lower slope and upper slope e tables
                    for action in state.upper_slope.e0:
                        state.upper_slope.e0[action] = 0
                    for action in state.upper_slope.e1:
                        state.upper_slope.e1[action] = 0
                    for action in state.upper_slope.e2:
                        state.upper_slope.e2[action] = 0

                    for action in state.lower_slope.e0:
                        state.lower_slope.e0[action] = 0
                    for action in state.lower_slope.e1:
                        state.lower_slope.e1[action] = 0
                    for action in state.lower_slope.e2:
                        state.lower_slope.e2[action] = 0
                else:
                    if state.state_type != Constants.block_state:
                        for action in state.e0:
                            state.e0[action] = 0
                        for action in state.e1:
                            state.e1[action] = 0
                        for action in state.e2:
                            state.e2[action] = 0

    def __select_action(self, state, greedy_selection=False):
        current_state = state

        if greedy_selection:
            # Take greedy action
            if self.agent.is_loaded is False and self.agent.is_unloaded is False:
                # Consider q0 table
                q_table = current_state.q0
                return max(q_table, key=q_table.get)
            elif self.agent.is_loaded is True and self.agent.is_unloaded is False:
                # Consider q1 table
                q_table = current_state.q1
                return max(q_table, key=q_table.get)
            elif self.agent.is_loaded is True and self.agent.is_unloaded is True:
                # Consider q2 table
                q_table = current_state.q2
                return max(q_table, key=q_table.get)

        dice = random.uniform(0, 1)
        if dice < self.epsilon:
            # Take random action
            if current_state.state_type != Constants.load_state and current_state.state_type != Constants.unload_state:
                return random.choice(current_state.get_possible_actions())
            elif current_state.state_type == Constants.load_state:
                return random.choice(current_state.get_possible_actions(is_loaded=self.agent.is_loaded,
                                                                        is_unloaded=self.agent.is_unloaded))
            elif current_state.state_type == Constants.unload_state:
                return random.choice(current_state.get_possible_actions(is_loaded=self.agent.is_loaded,
                                                                        is_unloaded=self.agent.is_unloaded))
        else:
            # Take greedy action
            if self.agent.is_loaded is False and self.agent.is_unloaded is False:
                # Consider q0 table
                q_table = current_state.q0
                return max(q_table, key=q_table.get)
            elif self.agent.is_loaded is True and self.agent.is_unloaded is False:
                # Consider q1 table
                q_table = current_state.q1
                return max(q_table, key=q_table.get)
            elif self.agent.is_loaded is True and self.agent.is_unloaded is True:
                # Consider q2 table
                q_table = current_state.q2
                return max(q_table, key=q_table.get)

    def __take_action(self, state, action):
        next_state = None
        reward = None
        is_episode_finished = None

        current_state = state
        current_x = current_state.coords[0]
        current_y = current_state.coords[1]


        # If the agent in upper slope
        if current_state.state_type == Constants.upper_slope:
            # If action is right action, it can move to right state with 0.5 probability
            if action == Constants.right_action:
                dice = random.uniform(0, 1)
                if dice < 0.5:
                    # Failure
                    next_state = current_state
                    reward = -0.5
                    is_episode_finished = False
                    return next_state, reward, is_episode_finished
                else:
                    # Success
                    next_state = self.__determine_next_state(state=current_state, action=action)
                    reward = self.__determine_reward(state=next_state)
                    is_episode_finished = False
                    return next_state, reward, is_episode_finished
            # If action is up action, it will move to upper state
            elif action == Constants.up_action:
                next_state = self.__determine_next_state(state=current_state, action=action)
                reward = self.__determine_reward(state=next_state)
                is_episode_finished = False
                return next_state, reward, is_episode_finished
            # If action is done action, it will stay at the current state
            elif action == Constants.down_action:
                next_state = current_state
                reward = self.__determine_reward(state=next_state)
                is_episode_finished = False
                return next_state, reward, is_episode_finished
        # If the agent in lower slope
        elif current_state.state_type == Constants.lower_slope:
            # If action is left action, it can move to left state with 0.5 probability
            if action == Constants.left_action:
                dice = random.uniform(0, 1)
                if dice < 0.5:
                    # Failure
                    next_state = current_state
                    reward = -0.5
                    is_episode_finished = False
                    return next_state, reward, is_episode_finished
                else:
                    # Success
                    next_state = self.__determine_next_state(state=current_state, action=action)
                    reward = self.__determine_reward(state=next_state)
                    is_episode_finished = False
                    return next_state, reward, is_episode_finished
            # If action is up action, it will stay at the current state
            elif action == Constants.up_action:
                next_state = current_state
                reward = self.__determine_reward(state=next_state)
                is_episode_finished = False
                return next_state, reward, is_episode_finished
            # If action is done action, it will move to lower state
            elif action == Constants.down_action:
                next_state = self.__determine_next_state(state=current_state, action=action)
                reward = self.__determine_reward(state=next_state)
                is_episode_finished = False
                return next_state, reward, is_episode_finished

        # If the current state is LoadState and agent takes load action, it stays at the current state
        if current_state.state_type == Constants.load_state and action == Constants.load_action:
            next_state = current_state
            reward = self.__determine_reward(state=next_state)
            is_episode_finished = False
            return next_state, reward, is_episode_finished

        # If the current state is UnloadState and agent takes unload action, it stays at the current state
        if current_state.state_type == Constants.unload_state and action == Constants.unload_action:
            next_state = current_state
            reward = self.__determine_reward(state=next_state)
            is_episode_finished = False
            return next_state, reward, is_episode_finished

        next_state = self.__determine_next_state(state=current_state, action=action)

        # If next_state is slope, determine whether it is upper slope or lower slope
        if next_state.state_type == Constants.slope_state:
            # If agent comes that state with either left or down action, means that it is upper slope
            if action == Constants.left_action or action == Constants.down_action:
                next_state = next_state.upper_slope
            # If agent comes that state with either right or up state, means that it is lower slope
            if action == Constants.right_action or action == Constants.up_action:
                next_state = next_state.lower_slope

        reward = self.__determine_reward(state=next_state)

        if reward == 1000:
            is_episode_finished = True
        else:
            is_episode_finished = False

        return next_state, reward, is_episode_finished

    def __determine_next_state(self, state, action):
        if action == Constants.load_action or action == Constants.unload_action:
            return state

        coord_x = state.coords[0]
        coord_y = state.coords[1]

        coord_new_x = None
        coord_new_y = None

        if action == Constants.left_action:
            coord_new_x = coord_x
            coord_new_y = coord_y - 1
        elif action == Constants.right_action:
            coord_new_x = coord_x
            coord_new_y = coord_y + 1
        elif action == Constants.up_action:
            coord_new_x = coord_x - 1
            coord_new_y = coord_y
        elif action == Constants.down_action:
            coord_new_x = coord_x + 1
            coord_new_y = coord_y

        # If new coords are out of bound, return state
        if coord_new_x < 0 or coord_new_x >= self.environment.n:
            return state
        elif coord_new_y < 0 or coord_new_y >= self.environment.n:
            return state

        # If new coords are blocking state, return state
        if self.environment.env[coord_new_x][coord_new_y].state_type == Constants.block_state:
            return state

        return self.environment.env[coord_new_x][coord_new_y]

    def __determine_reward(self, state):
        if state.state_type == Constants.rough_state:
            return -10

        if self.agent.is_loaded is True and self.agent.is_unloaded is True:
            if state.coords == (0, 0):
                return 1000

        return -0.1