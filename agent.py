from board import *
import numpy as np
import queue
import seaborn as sns
import matplotlib.pyplot as plt
q = queue.Queue(maxsize=20)

from queue import PriorityQueue

class Agent:
    def __init__(self, exploration_rate, learning_rate, n_steps=4, theta=0):
        self.board = Board()
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.x = self.board.x
        self.y = self.board.y
        self.actions = ["n", "s", "w", "e"]
        self.heat_map_number_of_pass = np.zeros([self.board.rows,self.board.columns])

        self.states = []
        # Dictionary
        self.state_actions = {}

        self.n_steps = n_steps
        
        self.model = {}
        
        for i in range(len(self.board.board)):
            for j in range(len(self.board.board[0])):
                self.state_actions[(i, j)] = {}
                for k in self.actions:
                    self.state_actions[(i, j)][k] = 0

        #Priority Sweeping
        self.theta = theta
        self.queue = PriorityQueue()
        self.predecessors = {}
        
        self.steps_per_episode = []


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def reset(self):
        self.states = []
        self.board = Board()
        self.x = self.board.startx
        self.y = self.board.starty

    def decide_action(self):
        action = None
        threshold_reward = -1000
        if self.exploration_rate >= np.random.uniform(0, 1):
            return np.random.choice(self.actions)
        else:
            # Greedy
            # All actions have same value
            if len(set(self.state_actions[(self.x, self.y)].values())) == 1:
                action = np.random.choice(self.actions)
            else:
                for i in self.actions:
                    currentx = self.x
                    currenty = self.y
                    #print(currentx, currenty, i)
                    reward = self.state_actions[(currentx, currenty)][i]
                    if reward >= threshold_reward:
                        action = i
                        threshold_reward = reward
        return action

    def train(self, rounds):
        for i in range(rounds):
            self.queue = PriorityQueue()
            number_of_steps = 0
            # print('Round: ' + str(i))
            # sum_reward = 0
            # max_reward = -99999
            while not self.board.is_agent_reach:
                action = self.decide_action()
                current_state = (self.x, self.y)
                self.states.append((current_state, action))

                nxtState = self.board.move(action)

                self.heat_map_number_of_pass[nxtState] += 1
                
                self.board.x = nxtState[0]
                self.board.y = nxtState[1]
                self.x = self.board.x
                self.y = self.board.y
                
                reward = self.board.reward()    

                #print( self.state_actions )
                #print( np.max(list(self.state_actions[nxtState].values())) )
                tmp_diff = abs(reward + np.max(list(self.state_actions[nxtState].values())) - self.state_actions[current_state][action])
                if tmp_diff > self.theta:
                    try:
                        self.queue.put((tmp_diff, (current_state, action)))
                    except MemoryError:
                        print(MemoryError)
                        break

                # Update model and predecessors ????????????????
                if current_state not in self.model.keys():
                    self.model[current_state] = {}
                self.model[current_state][action] = (reward, nxtState)
                if nxtState not in self.predecessors.keys():
                    self.predecessors[nxtState] = [(current_state, action)]
                else:
                    self.predecessors[nxtState].append((current_state, action))
                current_state = nxtState
                

                

                for _ in range(self.n_steps):
                    if self.queue.empty():
                        break
                    _state, _action = self.queue.get()[1]
                    _reward, _nxtState = self.model[_state][_action]
                    self.state_actions[_state][_action] += self.learning_rate * (_reward + np.max(list(self.state_actions[_nxtState].values())) - self.state_actions[_state][_action])

                    # Loop in all states, action predicted lead to _state
                    if _state not in self.predecessors.keys():
                        continue
                    pre_state_action_list = self.predecessors[_state]

                    for (pre_state, pre_action) in pre_state_action_list:
                        pre_reward, _ = self.model[pre_state][pre_action]
                        pre_tmp_diff = abs(pre_reward + np.max(list(self.state_actions[_state].values())) - self.state_actions[pre_state][pre_action])
                        try:
                            if pre_tmp_diff > self.theta:
                                self.queue.put((pre_tmp_diff, (pre_state, pre_action)))
                        except MemoryError:
                            print(MemoryError)
                            break
                
                if self.exploration_rate > 0.005 and self.board.is_agent_reach:
                    self.exploration_rate -= 0.05
                if self.board.is_agent_die:
                    break
            print("#####", int(i), (self.exploration_rate))
            self.steps_per_episode.append(len(self.states))
            #print(self.steps_per_episode)
            
            
            if i > 10:
                sum_of_10 = 0
                for e in range(1, 11):
                    sum_of_10 += self.steps_per_episode[-e]
                    #print("@@@@", self.steps_per_episode[-e])

                mean_of_10 = sum_of_10/10

                sum_variance = 0
                for e in range(1, 11):
                    sum_variance += pow((self.steps_per_episode[-e] - mean_of_10),2)
                        
                variance = sum_variance / (10 - 1)
                if(variance == 0):
                    print("VARIANCE == 0")
                    break
                
                

            self.reset()


        
        # print (self.state_actions)

        sns.set()
        # nparray = np.array(self.steps_per_episode)
        # # print(nparray)
        # B = np.reshape(nparray, (-1, 25))
        print(self.heat_map_number_of_pass)
        ax = sns.heatmap(self.heat_map_number_of_pass)
        plt.imshow(self.heat_map_number_of_pass, cmap='hot', interpolation='nearest')
        plt.show()

        #         current_state = (self.x, self.y)
        #         current_reward = self.board.reward()
        #         action = self.decide_action()

        #         # Calculate actions' reward
        #         move_tuple = self.board.move(action)
        #         self.board.x = move_tuple[0]
        #         self.board.y = move_tuple[1]
        #         self.x = self.board.x
        #         self.y = self.board.y
                
                
        #         number_of_steps += 1

        #         self.states.append([current_state, action, current_reward])

        #         # Calculate total reward of the round
        #         sum_reward += sum_reward + self.board.reward()

        #         # Get maximum of the rounds' reward
        #         if sum_reward > max_reward:
        #             max_reward = sum_reward

        #         if self.board.is_agent_die or self.board.is_agent_reach:
        #             break
            
        #     if(q.full()):
        #         q.get()
        #     print(number_of_steps)
        #     q.put(int(number_of_steps))

        #     print("QUEUE:", q.queue[0])
            

        #     reward = self.board.reward()
        #     print("REWARD ", sum_reward)


        #     for j in self.actions:
        #         self.state_actions[(self.x, self.y)][j] = reward

        #     for s in reversed(self.states):
        #         position = s[0]
        #         action = s[1]
        #         r = s[2]
        #         current_value = self.state_actions[position][action]
        #         reward = current_value + self.learning_rate * (r + reward - current_value)
        #         self.state_actions[position][action] = round(reward, 3)
        #         reward = np.max(list(self.state_actions[position].values()))
            

        #     if self.exploration_rate > 0.005:
        #         self.exploration_rate -= 0.0001

            
        #     sum_of_queue = 0
        #     if(q.full()):
        #         for e in list(q.queue):
        #             sum_of_queue += e

        #         mean_of_queue = sum_of_queue / q.qsize()

        #         sum_variance = 0
        #         for e in list(q.queue):
        #             sum_variance += pow((e - mean_of_queue),2)
                    
        #         variance = sum_variance / (q.qsize() - 1)


        #         print(q.queue)
        #         print("MEAN:" , mean_of_queue)
        #         print("VARIANCE:" , variance)
        #         if(variance == 0):
        #             break

        #     self.reset()
        # print("Maximum Reward of the training of " + str(rounds) + " rounds :" + str(max_reward))

    def render(self, states):
        for i in range(0, len(self.board.board)):
            for j in range(0, len(self.board.board[0])):
                p = ' 0'
                if self.board.board[i, j] == -1:
                    p = ' X'
                if (i, j) in states:
                    p = ' #'
                if i == self.board.endx and j == self.board.endy:
                    p = ' E'
                if i == self.board.startx and j == self.board.starty:
                    p = ' S'
                print(p, end='')
            print()