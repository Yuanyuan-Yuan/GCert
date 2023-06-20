# GenProver

The official implementation of GenProver is provided [here](https://openreview.net/forum?id=HJxRMlrtPH).

After downloading the code, you need to modify the following scripts in the projects:

- `components.py` - GenProver is implemented based on [DiffAI](https://github.com/eth-sri/diffai) and `components.py` re-implements different Pytorch `nn` modules with `InferModule` of DiffAI. We modified the implementations of several modules (mostly the `BatchNorm` module) to better fit  the implementations in Pytorch. You can replace the original `components.py` with our provided one.

- `genmodels.py` - We added implementations (with DiffAI modules) of our models in this script. You can replace the original `genmodels.py` with our provided one.

Note that in order to load models trained with Pytorch, you need to do the following:

1. Implement the model following the examples given in `experiments/model.py`. We suggest implementing the model with `nn.Sequential()` and hard-coding the name for each `nn.Sequential()`.

2. Implement every operation as a class inherited from Pytorch `nn` module. For example, the `torch.cat()` operation should be implement as `class CatTwo(nn.Module)`; see examples in `experiments/model.py`.

3. Implement the corresponding class following DiffAI in `components.py`. For example, for the `class CatTwo(nn.Module)` in `experiments/model.py`, you should implement a `class CatTwo(InferModule)` in `components.py`; more examples are given in `components.py`.

4. When loading the trained weights, you need to convert the key in `state_dict`. We provide the implementation and examples in `load_model.py`.