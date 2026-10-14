from abc import ABC, abstractmethod
from typing import Dict, Union, Any
import numpy as np

class BaseDepthEstimator(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def infer(self,
              img: np.ndarray,
              mask: np.ndarray = None,
              **kwargs) -> Dict[str, Union[np.ndarray, Any]]:
        """Run inference on an image.

        Args:
        ----
            img: The image loaded as a numpy array.

        Returns:
        -------
            prediction: A dictionary containing the following keys:
                depth: estimated depth image
        """
        pass

    