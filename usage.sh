

python multiplayer.py \
    -agent_1 D3QN \
    -total_episode 400 \
    -memo_1 d3qn_huber_nstep \
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
    -agent_1 D3QN \
    -total_episode 400 \
    -memo_1 d3qn_huber_shaping \
    -run_root runs_bef_nstep/newer \
    --loss huber \
    --reward_shaping score_delta 

python multiplayer.py \
    -agent_1 D3QN \
    -total_episode 400 \
    -memo_1 d3qn_huber_nstep_shaping \
    -run_root runs_bef_nstep/newer \
    --loss huber \
    --n_step 3 \
    --reward_shaping score_delta 
