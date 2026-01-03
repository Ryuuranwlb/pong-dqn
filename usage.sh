# 用已经有的左手，训练左手 agent。注意需要 flip2 因为 agent2 的位置需要的其实是左手。。。很神奇吧

python multiplayer.py \
    -agent_1 D3QN \
    -total_episode 400 \
    -player 2 \
    -memo_1 l_d3qn_huber \
    -memo_2 l_d3qn_huber \
    -start_step_2 278953 \
    -agent_2 D3QN \
    -run_root final_submit \
    --loss huber \
    -train_left 


# 用已经有的左手，训练右手 agent。

python multiplayer.py \
    -agent_1 D3QN \
    -total_episode 400 \
    -player 2 \
    -memo_1 r_d3qn_huber \
    -memo_2 l_d3qn_huber \
    -start_step_2 278953 \
    -agent_2 D3QN \
    -run_root final_submit \
    --loss huber \