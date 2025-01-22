# 世界モデルにおける時空間情報の理解に関する検証 / Investigating the Understanding of Spatio-temporal Information in World Models


## Getting Started
1. After cloning this repo, clone DreamerV3 repository
- [DreamerV3](https://github.com/danijar/dreamerv3)

2. Replace the following file paths in the DreamerV3 repository with files in our dreamerv3_edits directory.
- /dreamerv3/configs.yaml
- /embodied/envs/crafter.py

3. execute the command `./training_dreamer.sh` to train DreamerV3.

4. After training DreamerV3, run probing/save_activations.ipynb to collect activations.

5. Finally, probing is performed by running probing/probing.ipynb.
