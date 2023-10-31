# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Explain the problem statement.

## SARSA LEARNING ALGORITHM
![Uploading Screenshot 2023-10-31 093306.png…]()

## SARSA LEARNING FUNCTION
```python
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)

    epsilons=decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        next_state,reward,done,_=env.step(action)
        next_action=select_action(next_state,Q,epsilons[e])
        td_target=reward+gamma*Q[next_state][next_action]*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state,action=next_state,next_action
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
The optimal policy, optimal value function , success rate for the optimal policy.
![Screenshot 2023-10-31 092746](https://github.com/Dineshkumar021617/sarsa-learning/assets/75234807/6af661af-0043-4e82-bfa7-53163bab93ce)


Plot comparing the state value functions of Monte Carlo method and SARSA learning.
![image](https://github.com/Dineshkumar021617/sarsa-learning/assets/75234807/47170090-58a0-43f7-9483-4df0767d4c72)

![image](https://github.com/Dineshkumar021617/sarsa-learning/assets/75234807/82802720-50db-4247-baeb-54f9fbd3e1fa)

## RESULT:

Thus, a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method is developed and implemented successfully.
