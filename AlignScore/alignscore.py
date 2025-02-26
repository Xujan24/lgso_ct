from .inference import Inferencer
from typing import List
import torch

class AlignScore:
    def __init__(self, model: str = 'roberta-large', batch_size: int = 32, ckpt_path: str = 'AlignScore/checkpts/AlignScore-large.ckpt', evaluation_mode='nli_sp', verbose=False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Inferencer(
            ckpt_path=ckpt_path, 
            model=model,
            batch_size=batch_size, 
            device=self.device,
            verbose=verbose
        )
        self.model.nlg_eval_mode = evaluation_mode

    def score(self, contexts: List[str], claims: List[str]) -> List[float]:
        return self.model.nlg_eval(contexts, claims)[1].tolist()