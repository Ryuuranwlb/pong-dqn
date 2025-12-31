# RLPong

高级强化学习——乒乓球对战大作业

## 环境配置

根据gym-retro仓库的说法，该库最高支持python3.8，因此本项目使用python3.8作为编译器。

主要依赖库可以从仓库根目录下的`requirement.txt`了解。

此外，由于代码`multiplayer.py`中还用到了`FFMpegWriter`，因此还需要安装`FFMpeg`相关的支撑软件（这些软件并不是python的包），可通过如下方式安装：

```bash
conda install -c conda-forge ffmpeg
```

为了能够使用`Pong-Atari2600`这个游戏环境，我们还需要额外安装相应的Roms。安装的具体步骤请参考[issue](https://github.com/openai/retro/issues/60)

### 具体流程

1. 配置Python环境

```bash
conda create --name rlp python=3.8

conda activate rlp
```

2. 安装gym

```bash
pip install gym==0.23.1
```

3. 安装`gym-retro`

```bash
pip install gym-retro==0.8.0
```

4. 安装`ffmpeg`

```bash
conda install -c conda-forge ffmpeg
```

5. 安装`Pong-Atari2600`的Rom

压缩包已给出

解压压缩包到指定文件夹，本人是解压到了项目根目录下的`./Roms`文件夹下

执行命令

```bash
python -m retro.import ./Roms
```

> 配置完毕