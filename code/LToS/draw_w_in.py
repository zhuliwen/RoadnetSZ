import numpy as np
import matplotlib.pyplot as plt

num_episodes = 100
num_generators = 4
num_rows, num_cols = 6, 6
num_agents = num_rows*num_cols
len_episode = 3600
interval = 10
total_w_in = np.zeros((num_episodes*num_generators,num_agents,len_episode//interval))
with open('w_in_test.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        line = line.strip()

        time, w_in = line.split(':')
        time = int(float(time)) // interval
        w_in = [float(s) for s in w_in.split(',')]
        
        episode = i // (len_episode//interval * num_agents)
        index = (i // (len_episode//interval)) % num_agents
        r, c = index//num_rows, index%num_rows
        if r == 0 and c == 0:
            total_w_in[episode][index][time] = w_in[0]
        elif r == 0 or c == 0:
            total_w_in[episode][index][time] = w_in[1]
        else:
            total_w_in[episode][index][time] = w_in[2]

for episode in range(32,100):
    for i in range(num_agents):
        print(total_w_in[episode][i])
        r, c = i//num_rows, i%num_rows
        if r == 0 or c == 0:
            plt.plot(range(len(total_w_in[episode][i])), total_w_in[episode][i], label=i)
    plt.title('Episode: %d' % episode)
    plt.yticks(np.arange(0.75, 1, 0.05))
    plt.legend(loc='upper left')
    plt.show()