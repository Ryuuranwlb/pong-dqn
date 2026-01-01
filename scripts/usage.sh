

python multiplayer.py \
    -agent_1 D3QN \
    -total_episode 400 \
    -memo_1 d3qn_huber_nstep \
    -run_root runs_bef_nstep/newer \
    --loss huber \
    --n_step 3

python multiplayer.py \
    -agent_1 DoubleDQN \
    -total_episode 400 \
    -memo_1 doubledqn_huber_nstep \
    -run_root runs_bef_nstep/newer \
    --loss huber \
    --n_step 3

python multiplayer.py \
    -agent_1 DuelingDQN \
    -total_episode 400 \
    -memo_1 duelingdqn_huber_nstep \
    -run_root runs_bef_nstep/newer \
    --loss huber \
    --n_step 3

python multiplayer.py \
    -agent_1 DQN \
    -total_episode 400 \
    -memo_1 dqn_huber_shaping \
    -run_root runs_bef_nstep/newer \
    --loss huber \
    --reward_shaping score_delta 

python multiplayer.py \
    -agent_1 DQN \
    -total_episode 400 \
    -memo_1 dqn_mse\
    -run_root runs_bef_nstep/newer \
    --loss mse 
