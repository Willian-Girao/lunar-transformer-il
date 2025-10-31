# Learning to Land: Imitation Learning with a Decoder-Only Transformer

This project explores how a **decoder-only Transformer** can learn to control the [Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment purely through **imitation learning** (IL) from an expert trained with Actor–Critic reinforcement learning (RL).

The trained Transformer is able to **land the spacecraft successfully**, and even **generalizes** when the environment physics (e.g., gravity) are modified!

![Landing Demo](results/media/lander_success.gif)

## Motivation

Reinforcement learning agents are often **sample-inefficient** and **sensitive to environment changes**.  
Here, I investigate whether a Transformer trained purely from **expert demonstrations** can:
- Achieve competent control performance; and  
- **Generalize** better than the expert when faced with altered physics.

## Approach Overview

The project pipeline consists of four main stages:

<table>
<tr>
<td>

### 1. Expert Acquisition :brain:
- A pretrained (on the standard Lunar Lander environment) [actor–critic agent](https://github.com/nikhilbarhate99/Actor-Critic-PyTorch) is used.
- This agent achieves stable landings and serves as the **expert**.

### 2. Data Generation :rocket:
- The expert policy produces 1000 trajectories of `(state, action)` tuples plus the accumulated reward per episode.
- These rollouts are padded to get fixed-length episodes, stored and later tokenized into sequences for the Transformer.
- The normalized accumulated rewards are used to weight the losses during training.

### 3. Transformer Training :robot:
- A **decoder-only Transformer** learns to predict the next action given the previous states and actions.
- Coarse search determines reasonable architectural bounds (e.g., depth, heads, embedding dimension).
- **Hyperparameter optimization (HPO)** is then performed using [NNI](https://nni.readthedocs.io/).

### 4. Evaluation & Generalization :video_game:
- The trained model is compared with the expert in:
  - The **original environment**, and  
  - **Altered-gravity** variants (±10%).

</td>
<td>

<img src="data/media/pipeline.svg" alt="Pipeline Diagram" width="250">

</td>
</tr>
</table>