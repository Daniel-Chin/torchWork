# torchWork
A framework / toolkit for deep learning projects using pyTorch.  

Features: 
- Losses as trees.  
- Loss logging & plotting.  
- Keeping track of hyper parameters.  
- Experiment management.  
  - Vary hyper parameters to generate experiment groups.  
  - Archive hyper parameters & git commit hash for reproducibility.  
  - Time-base round robin across experiment groups.  
- Profiler.  

## Example usage
...

## Features
### Losses as trees
Meta programming. Reasons...  
Sum using weights.  
Add two trees to get another tree.  

### Loss logging & plotting
- Log the losses as trees in .txt files.  
- Plot the losses.  
- You can log extra info such as grad_norm.  
