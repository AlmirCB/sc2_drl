# sc2_drl

This repository belongs to the development of the master's thesis "Reducing Computational Costs in Deep Reinforcement Learning for Real-Time Strategy Games."

The code has been developed to demonstrate that breaking down a complex problem into groups of simpler tasks can reduce the computational cost of training a deep reinforcement learning solution. The video game Starcraft II has been used as the environment for this purpose.
![alt text](https://bnetcmsus-a.akamaihd.net/cms/blog_header/2g/2G4VZH5TIWJF1602720144046.jpg)

## How does it works?
To train an agent, you need to create a game configuration in the "game_configs.py" file and select the appropriate agent from all available ones. Finally, you need to load both components in the "train.py" file and run the code:
```bash
python train.py --playing=False
```
