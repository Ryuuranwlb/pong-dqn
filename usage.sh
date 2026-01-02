

python multiplayer.py \
    -agent_1 D3QN \
    -total_episode 400 \
    -player 2 \
    -memo_1 l_d3qn_huber_default \
    -memo_2 r_teacher_ori \
    -start_step_2 23333 \
    -run_root final_submit \
    --loss huber \
    --train_left
