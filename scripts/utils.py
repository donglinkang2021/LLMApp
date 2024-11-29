from dataclasses import dataclass
from typing import List, Optional, Optional, Union
import numpy as np

@dataclass
class Response:
    model: str
    created_at: str
    response: str
    done: bool
    done_reason: Optional[str] = None
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

    @staticmethod
    def from_dict(data: dict) -> 'Response':
        return Response(
            model=data['model'],
            created_at=data['created_at'],
            response=data['response'],
            done=data['done'],
            done_reason=data.get('done_reason'),
            context=data.get('context'),
            total_duration=data.get('total_duration'),
            load_duration=data.get('load_duration'),
            prompt_eval_count=data.get('prompt_eval_count'),
            prompt_eval_duration=data.get('prompt_eval_duration'),
            eval_count=data.get('eval_count'),
            eval_duration=data.get('eval_duration')
        )


@dataclass
class ModelDetails:
    parent_model: str
    format: str
    family: str
    families: List[str]
    parameter_size: str
    quantization_level: str

@dataclass
class Model:
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: ModelDetails

    @staticmethod
    def from_dict(data: dict) -> 'Model':
        return Model(
            name=data['name'],
            model=data['model'],
            modified_at=data['modified_at'],
            size=data['size'],
            digest=data['digest'],
            details=ModelDetails(
                parent_model=data['details']['parent_model'],
                format=data['details']['format'],
                family=data['details']['family'],
                families=data['details']['families'],
                parameter_size=data['details']['parameter_size'],
                quantization_level=data['details']['quantization_level']
            )
        )

@dataclass
class EmbeddingResponse:
    model: str
    embeddings: np.ndarray[np.float32]
    total_duration: int = None
    load_duration: int = None
    prompt_eval_count: int = None

    @staticmethod
    def from_dict(data: dict) -> 'EmbeddingResponse':
        return EmbeddingResponse(
            model=data['model'],
            embeddings=np.array(data['embeddings'], dtype=np.float32), # List[List[float]] -> np.ndarray[np.float32]
            total_duration=data.get('total_duration'),
            load_duration=data.get('load_duration'),
            prompt_eval_count=data.get('prompt_eval_count')
        )

def get_tokens_per_second(eval_count, eval_duration):
    """
    Params
    ----
    eval_count: int
        number of tokens in the prompt or response
    eval_duration: int
        time spent in nanoseconds evaluating the prompt or generating the response
    """
    return eval_count / eval_duration * 1e9

def show_reponse_metric(pack_data: Union[Response, EmbeddingResponse]):
    print(f"total duration: {pack_data.total_duration * 1e-9:.6f}s")
    print(f"loading the model duration: {pack_data.load_duration * 1e-9:.6f}s")
    if isinstance(pack_data, EmbeddingResponse):
        return
    tokens_per_second = get_tokens_per_second(
        pack_data.eval_count, 
        pack_data.eval_duration
    )
    prompt_tokens_per_second = get_tokens_per_second(
        pack_data.prompt_eval_count, 
        pack_data.prompt_eval_duration
    )
    print(f"evaluating the prompt speed: {tokens_per_second:.6f} tokens/s")
    print(f"generating the response speed: {prompt_tokens_per_second:.6f} tokens/s")
