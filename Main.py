from Environment import Environment
from Agent import Agent
from FlatQLearning import FlatQLearning
from LambdaQLearning import LambdaQLearning

# Get input for the environment size
env_size = int(input('Please enter environment size (nxn): '))
environment = Environment(n=env_size)

# Get will be used algorithm
algorithm_choice = int(input('Please select algorithm.\n1) Flat Q Learning\n2) Lambda Q-Learning\nChoice (1 or 2): '))

if algorithm_choice == 1:
    learning_rate = float(input('Please enter learning rate: '))
    discount_rate = float(input('Please enter discount rate: '))
    epsilon = float(input('Please enter epsilon value: '))
    flat_q_learning = FlatQLearning(environment=environment, agent=Agent(), learning_rate=learning_rate,
                                    discount_rate=discount_rate, epsilon=epsilon)
    # Train
    num_of_episodes = int(input('Please enter number of episodes for training process: '))
    flat_q_learning.train(num_of_episodes=num_of_episodes)

    test_choice = input('Do you want to test it? (y or n): ')
    if test_choice == 'y':
        flat_q_learning.test()
elif algorithm_choice == 2:
    learning_rate = float(input('Please enter learning rate: '))
    discount_rate = float(input('Please enter discount rate: '))
    epsilon = float(input('Please enter epsilon value: '))
    lambda_value = float(input('Please enter lambda value: '))

    lambda_q_learning = LambdaQLearning(environment=environment, agent=Agent(), learning_rate=learning_rate,
                                        discount_rate=discount_rate,
                                        epsilon=epsilon, lambda_value=lambda_value)

    # Train
    num_of_episodes = int(input('Please enter number of episodes for training process: '))
    lambda_q_learning.train(num_of_episodes=num_of_episodes)

    test_choice = input('Do you want to test it? (y or n): ')
    if test_choice == 'y':
        lambda_q_learning.test()

