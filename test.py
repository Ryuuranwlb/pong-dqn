import retro

def inspect_info_keys():
    game_name = 'Pong-Atari2600'
    
    try:
        # 1. 创建环境
        # 注意：需要先导入 ROM (python -m retro.import <path_to_roms>)
        env = retro.make(game=game_name)
        
        # 2. 重置环境 (必须在 step 之前调用)
        env.reset()
        
        # 3. 随机执行一个动作以获取第一帧的返回结果
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        
        # 4. 打印 info 中的 keys
        print(f"--- Game: {game_name} ---")
        print("Info 字典中的 Keys:")
        print(list(info.keys()))
        
        # 5. (可选) 打印具体的值以供参考
        print("\n当前值的示例:")
        print(info)
        
        env.close()
        
    except FileNotFoundError:
        print(f"错误: 找不到游戏 '{game_name}'。")
        print("请确保你已经拥有该游戏的 ROM 并使用 'python -m retro.import /path/to/rom' 进行了导入。")
    except Exception as e:
        print(f"发生了未预期的错误: {e}")

if __name__ == "__main__":
    inspect_info_keys()