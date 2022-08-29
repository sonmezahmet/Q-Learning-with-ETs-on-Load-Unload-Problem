# Q-Learning with ETs on Load Unload Problem
## Problem Definition
In this project, the problem is implementing the flat Q-learning algorithm and the Qlearning
algorithm with ETs which uses replacing traces for the environment. Also
environment should be created dynamically according to user’s input.
<br>
## Methodology
### Creating Environment
![image](https://user-images.githubusercontent.com/56430166/187248501-3805ab43-63ac-410a-bb9a-1bf5c571791c.png)
<br>
Before I started creating the environment, I divided the environment into different states. I
created 9 different classes for those states which are, ‘State’, ‘RegularState’, LoadState’,
‘UnloadState’, ‘RoughState’, ‘BlockState’, ’SlopeState’, ‘UpperSlope’ and ‘LowerSlope’.
‘State’ is base class for others. ‘RegularState’ presents normal states which can be
moveable. ‘RoughState’ presents rough roads on the grid world. ‘BlockState’ presents
can’t be moveable states. ‘SlopeState’ indicates that state is slope state, and it contains
2 other sub states which are ‘UpperSlope’ and ‘LowerSlope’. Each state has 3 different Q
tables and 3 different ET table. First Q table and ET table is used when agent isn’t loaded
yet, second Q table and ET table is used when agent is loaded and isn’t unloaded yet,
and last Q table and ET table is used when agent is unloaded.
Environment class has only one class parameter which is ’n’ for size of grid world. For ’n’,
minimum value is 6, e.g., if you try create 5x5 environment, it creates automatically 6x6
environment.
<br>
Environment class has 3 class functions which are ‘__create_2d_matrix' for creating n x n
matrix, ‘__create_environment’ for placing proper states to created n x n matrix,
‘print_environment’ for printing environment.
### Implementing Flat Q-Learning
FlatQLearning class has 5 different class attributes. ‘environment’ for keeping track of
environment, ‘agent’ for the agent, and hyper-parameters such as ‘learning_rate’,
‘discount_rate’, and ‘epsilon’.
<br>
In FlatQLearning class, I have 4 helper methods which are ‘__select_action’,
‘__take_action’, ‘__determine_next_state’, and ‘__determine_reward’. ‘__select_action’
method returns an action according the agent’s current state at the moment and it usese-greedy policy. ‘__take_action’ method performs an action and returns next state, reward
and information of whether episode is finished or not. ‘__determine_next_state’ and
‘__determine_reward’ methods are used in ‘__take_action’ method.
![image](https://user-images.githubusercontent.com/56430166/187246782-35e266f0-656c-4323-926f-35a736481eb2.png) <br>
In ‘train’ method, I implemented flat Q-learning algorithm with using helper methods
which I mentioned above.
![image](https://user-images.githubusercontent.com/56430166/187246886-9cbebeed-f772-48c0-a83b-c65ef189aa0d.png)
### Implementing Q-Learning with ETs (which uses replacing traces)
In LambdaQLearning class there are 6 class parameters which are ‘environment’, ‘agent’,
‘learning_rate’, ‘discount_rate’, ‘epsilon’, and ‘lambda_value’.
In LambdaQlearning class, I have 5 helper methods which are ‘__select_action’,
‘__take_action’, ‘__determine_next_state’, ‘__determine_reward’, and ‘__reset_e_values’.
‘__select_action’ method returns an action according the agent’s current state at the
moment and it uses e-greedy policy. ‘__take_action’ method performs an action and
returns next state, reward and information of whether episode is finished or not.
‘__determine_next_state’ and ‘__determine_reward’ methods are used in ‘__take_action’
method. ‘__reset_e_tables’ method initialize all states’ ET table to zero.
<br>
![image](https://user-images.githubusercontent.com/56430166/187247022-3deacba4-d951-4e21-ac4e-e3887351261f.png) <br>
In ‘train’ method, I implemented Watkin’s Q-learning algorithm with replacing traces.
![image](https://user-images.githubusercontent.com/56430166/187247107-d124dd32-833d-43b0-9ce4-5910c1080e7f.png)
## Experiments and Results
I run two algorithms for 300 episodes with learning rate is 0.3, discount rate is 0.9, epsilon
is 0.08, and for lambda Q-learning, lambda is 0.3 in 10x10 environment. Results are shown in below. <br>
![image](https://user-images.githubusercontent.com/56430166/187247263-e13d2d89-9796-4d0a-8ee3-5a9dc28a8ed2.png) <br>
![image](https://user-images.githubusercontent.com/56430166/187247289-72113be8-f08a-4092-b5fd-5dd5b95669e4.png) <br>
## Conclusion
As seen in the experiments, although Lambda Q-Learning reaches large steps in the first
episodes, it converges to the optimum policy faster than Flat Q-Learning.
Although Lambda Q-Learning is more difficult to implement than Flat Q-Learning, it can
perform faster and more efficient learning in more complex environments.
Apart from these, I did not use replacing traces at first when implementing Lambda QLearning.
As a result, the training process became extremely slow (even after waiting for a
long time, I could not see the training results), but the performance of this algorithm
increased noticeably when I used replacing traces.



