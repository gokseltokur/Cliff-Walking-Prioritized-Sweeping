from board import *
from agent import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

EXPLORATION_RATE = 0.4
LEARNING_RATE = 0.1
NUMBER_OF_ROUNDS = 10000


def main():
    agent_train = Agent(EXPLORATION_RATE, LEARNING_RATE)
    agent_train.train(NUMBER_OF_ROUNDS)

    agent = Agent(0, LEARNING_RATE)
    agent.state_actions = agent_train.state_actions

    states = []
    while 1:
        current_state = (agent.x, agent.y)
        action = agent.decide_action()
        states.append(current_state)
        
        print("POSITION: " + str(current_state), end=' ')
        print("ACTION: " + str(action))


        move_tuple = agent.board.move(action)
        agent.board.x = move_tuple[0]
        agent.board.y = move_tuple[1]
        agent.x = agent.board.x
        agent.y = agent.board.y

        
        agent.render(states)

        if agent.board.is_agent_reach:
            break

    print(agent.state_actions)



    for i in range(len(agent.board.board)):
        for j in range(len(agent.board.board[0])):
            agent.heat_map_q_values[i][j] = max(agent.state_actions[(i, j)].values())

    sns.set()
    # nparray = np.array(self.steps_per_episode)
    # # print(nparray)
    # B = np.reshape(nparray, (-1, 25))
    print(agent.heat_map_q_values)
    ax = sns.heatmap(agent.heat_map_q_values, vmin=0, vmax=0.0000002)
    plt.imshow(agent.heat_map_q_values, cmap='hot', interpolation='nearest')
    plt.show()




if __name__ == "__main__":
    main()