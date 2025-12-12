"""
NanoMAD-ML: CNN-based MAD fitting for BCDI analysis of core-shell nanoparticles.

Main modules:
- core_shell: Physics engine for particle generation and diffraction
- mad_model: U-Net CNN architecture
- mad_loss: Physics-informed loss function
- train: Training pipeline
- inference: 2D/3D inference with post-processing
- evaluate: Evaluation metrics and visualization
- generate_data: Synthetic training data generation
"""

__version__ = "0.1.0"
