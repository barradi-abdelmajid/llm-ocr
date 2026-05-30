"""GPU memory management utilities for model swapping between phases.

Usage:
  from gpu_utils import free_gpu_memory, get_gpu_memory

  # Before loading GLM-OCR
  free_gpu_memory()

  # After GLM-OCR, before loading LM model
  free_gpu_memory()
"""

import time
import warnings


def get_gpu_memory() -> dict:
    """Return GPU memory info: total, used, free in MB."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mb = info.total / 1024 / 1024
        free_mb = info.free / 1024 / 1024
        used_mb = info.used / 1024 / 1024
        return {"total_mb": total_mb, "used_mb": used_mb, "free_mb": free_mb}
    except ImportError:
        return {"error": "pynvml not installed"}
    except Exception as e:
        return {"error": str(e)}


def free_gpu_memory(verbose: bool = True) -> None:
    """Aggressively free GPU memory: collect garbage, empty CUDA cache, synchronize."""
    import gc
    import torch

    if not torch.cuda.is_available():
        return

    if verbose:
        try:
            before = get_gpu_memory()
        except Exception:
            before = {}

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Also clear any cached tensors from transformers
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == "cuda":
                del obj
        except Exception:
            pass

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if verbose:
        try:
            after = get_gpu_memory()
            if before and "used_mb" in before and "used_mb" in after:
                freed = before["used_mb"] - after["used_mb"]
                if freed > 0:
                    print(f"  GPU: freed {freed:.0f}MB ({after['free_mb']:.0f}MB free)")
                else:
                    print(f"  GPU: {after['free_mb']:.0f}MB free")
            elif after and "free_mb" in after:
                print(f"  GPU: {after['free_mb']:.0f}MB free")
        except Exception:
            pass


def load_hf_model(
    model_id: str, task_type: str = "causal_lm", load_in_4bit: bool = False, **kwargs
):
    """Load a HuggingFace model on GPU, freeing memory first.

    Args:
        model_id: HuggingFace model ID
        task_type: "causal_lm" (text gen) or "vision_lm" (VLM)
        load_in_4bit: enable 4-bit quantization (requires bitsandbytes)
        **kwargs: passed to from_pretrained

    Returns: (model, processor/tokenizer)
    """
    free_gpu_memory(verbose=True)

    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )
    from transformers import BitsAndBytesConfig

    quant_kwargs = {}
    if load_in_4bit:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=kwargs.pop("bnb_4bit_compute_dtype", None)
            or "float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if task_type == "vision_lm":
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, **kwargs
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            **quant_kwargs,
            **kwargs,
        )
        model.eval()
        return model, processor
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, **kwargs
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            **quant_kwargs,
            **kwargs,
        )
        model.eval()
        return model, tokenizer


def unload_model(model, processor=None):
    """Unload a HuggingFace model from GPU memory."""
    import gc
    import torch

    if processor is not None:
        del processor
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
