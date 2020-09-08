# Cliff Walking Prioritized Sweeping
 Cliff Walking problem solution with Prioritized Sweeping algorithm
 
 This is a standard undiscounted, episodic task, with start and end state, and the usual actions causing movement north (n), south (s), east (e), and west (w). Reward is 0 on all transitions except those into the region marked “The Cliff” Stepping into this region incurs a reward of -10 and sends the agent instantly back to the start. If the agent reachs the end get +1 reward. 

 As the frontier of useful updates propagates backward, it often grows rapidly, producing
many state–action pairs that could usefully be updated. But not all of these will be
equally useful. The values of some states may have changed a lot, whereas others may
have changed little. The predecessor pairs of those that have changed a lot are more
likely to also change a lot. In a stochastic environment, variations in estimated transition
probabilities also contribute to variations in the sizes of changes and in the urgency with
which pairs need to be updated. It is natural to prioritize the updates according to a
measure of their urgency, and perform them in order of priority.

 A queue is maintained of every state–action pair whose estimated
value would change nontrivially if updated , prioritized by the size of the change. When
the top pair in the queue is updated, the e↵ect on each of its predecessor pairs is computed.
If the effect is greater than some small threshold, then the pair is inserted in the queue
with the new priority. In this way the effects of changes are efficiently propagated backward until quiescence.

## Prioritized Sweeping

![PrioritizedSweepingPsudo](https://github.com/gokseltokur/Cliff-Walking-Prioritized-Sweeping/blob/master/png/PrioritizedSweepingPsudo.png)


### Notation
Symbol  | Description
------------- | -------------
Q  | Value
S  | State
A  | Action
α  | Learning Rate
r  | Reward
γ  | Discount Factor
θ  | Theta
t  | Represents the time e.g. (t+1) -> future / t -> old  

## Example output of the 500 rounds of training with exploration rate of 0.2 and learning rate of 0.1 and theta 0

S -> Start point<br />
E -> End point (Goal)<br />
0 -> Plain<br />
X -> Cliff<br />
``#`` -> Path that agent went<br />

![Prioritized Sweeping Example Output](https://github.com/gokseltokur/Cliff-Walking-Prioritized-Sweeping/blob/master/png/Prioritized%20Sweeping%20Example%20Output.png)

## To Run
You can directly run the __train.py__
`python train.py`

## References
Richard S. Sutton and Andrew G. Barto, Reinforcement Learning: An Introduction 2018 (pp. 169, 170).


