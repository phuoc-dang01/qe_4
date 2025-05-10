import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import neat
import numpy as np
import torch

sys.path.append('/home/pd468/qe/evogym/examples/')
sys.path.append('/home/pd468/qe/evogym/examples/externals/PyTorch-NEAT')

from pytorch_neat.cppn import create_cppn


class CPPNRobotGenerator:
    @staticmethod
    def get_cppn_input(
        structure_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = torch.meshgrid(
            torch.arange(structure_shape[0]),
            torch.arange(structure_shape[1]),
            indexing="ij"
        )
        x, y = x.flatten(), y.flatten()
        center = (np.array(structure_shape) - 1) / 2
        d = ((x - center[0]) ** 2 + (y - center[1]) ** 2).sqrt()
        return x, y, d

    @staticmethod
    def get_robot_from_genome(
        genome: neat.DefaultGenome,
        config: neat.Config
    ) -> np.ndarray:
        nodes = create_cppn(
            genome,
            config,
            leaf_names=["x", "y", "d"],
            node_names=["empty", "rigid", "soft", "hori", "vert"],
        )
        structure_shape = config.extra_info["structure_shape"]
        x, y, d = CPPNRobotGenerator.get_cppn_input(structure_shape)
        material = []
        for node in nodes:
            material.append(node(x=x, y=y, d=d).numpy())
        material = np.vstack(material).argmax(axis=0)
        return material.reshape(structure_shape)
