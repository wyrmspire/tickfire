"""
Model registry.

This will map (model_family, config) -> constructed PyTorch model, e.g.:

- "gru": simple stacked GRU for sequence-to-vector outputs
- "gru_cond": GRU conditioned on macro scenario vectors
- "transformer": sequence models with attention

Later we'll add:
- build_model(cfg: ExperimentConfig) -> torch.nn.Module
- count_parameters(model) -> int
"""
