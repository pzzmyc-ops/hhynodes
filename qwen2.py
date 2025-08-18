import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def get_supported_float8_types():
    """Safe detection of supported fp8 types, similar to ComfyUI's approach"""
    float8_types = {}
    try:
        float8_types["fp8_e4m3fn"] = torch.float8_e4m3fn
    except:
        pass
    try:
        float8_types["fp8_e5m2"] = torch.float8_e5m2
    except:
        pass
    try:
        float8_types["fp8_e4m3fnuz"] = torch.float8_e4m3fnuz
    except:
        pass
    try:
        float8_types["fp8_e5m2fnuz"] = torch.float8_e5m2fnuz
    except:
        pass
    return float8_types

SUPPORTED_FP8_TYPES = get_supported_float8_types()

class Qwen25Inferencer:
    def __init__(self):
        self.device = None
        self.model_loaded = False
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        self.current_torch_dtype = None
        self.current_attn_implementation = None
        
    def set_device(self, device_type="auto"):
        if device_type == "auto":
            if torch.cuda.is_available():
                self.device = "cuda:0"
                self.device_map = {"": 0}
            else:
                self.device = "cpu"
                self.device_map = "cpu"
        elif device_type == "gpu":
            if torch.cuda.is_available():
                self.device = "cuda:0"
                self.device_map = {"": 0}
            else:
                print("GPU not available, falling back to CPU")
                self.device = "cpu"
                self.device_map = "cpu"
        else:  # cpu
            self.device = "cpu"
            self.device_map = "cpu"

    def get_torch_dtype(self, dtype_str):
        """Convert string to torch dtype"""
        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        
        # Add supported fp8 types
        dtype_map.update(SUPPORTED_FP8_TYPES)
        
        requested_dtype = dtype_map.get(dtype_str)
        
        if requested_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
            
        return requested_dtype

    def load_model(self, model_path, device_type="auto", torch_dtype="auto", attn_implementation="default", quantization="auto"):
        # Check if we need to reload the model
        need_reload = (
            not self.model_loaded or 
            self.current_model_path != model_path or
            self.current_torch_dtype != torch_dtype or
            self.current_attn_implementation != attn_implementation
        )
        
        if need_reload:
            # Unload current model if loaded
            if self.model_loaded:
                self.unload_model()
                
            self.set_device(device_type)
            print(f"Loading Qwen2.5 model from {model_path} on {self.device}...")
            print(f"Using torch_dtype: {torch_dtype}, attn_implementation: {attn_implementation}, quantization: {quantization}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Prepare model loading arguments
            model_kwargs = {
                "torch_dtype": self.get_torch_dtype(torch_dtype),
                "device_map": self.device_map
            }
            
            # Add attention implementation if not default
            if attn_implementation != "default":
                model_kwargs["attn_implementation"] = attn_implementation

            # Configure quantization
            if quantization != "auto":
                quantization_config = None
                if quantization == "int8":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                elif quantization == "int4":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.get_torch_dtype(torch_dtype),
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif quantization == "fp16":
                    model_kwargs["torch_dtype"] = torch.float16
                elif quantization == "bf16":
                    model_kwargs["torch_dtype"] = torch.bfloat16
                
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # Debug information
            print(f"Model loaded successfully!")
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Model dtype: {next(self.model.parameters()).dtype}")
            if hasattr(self.model.config, 'torch_dtype'):
                print(f"Model config dtype: {self.model.config.torch_dtype}")
            
            self.model_loaded = True
            self.current_model_path = model_path
            self.current_torch_dtype = torch_dtype
            self.current_attn_implementation = attn_implementation

    def unload_model(self):
        if self.model_loaded:
            print(f"Unloading model from {self.device}...")
            if self.device == "cuda:0":
                self.model = self.model.to("cpu")
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model_loaded = False
            self.model = None
            self.tokenizer = None
            self.current_model_path = None
            self.current_torch_dtype = None
            self.current_attn_implementation = None

    def generate_text(self, user_prompt, model_path, device_type="auto", torch_dtype="auto", attn_implementation="default", unload_model=False, max_new_tokens=32768, enable_thinking=True, quantization="auto"):
        try:
            self.load_model(model_path, device_type, torch_dtype, attn_implementation, quantization)
            
            # Prepare the messages
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            
            with torch.no_grad():
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                # Generate text
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens
                )
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

                # Parse thinking content and regular content
                thinking_content = ""
                content = ""
                
                if enable_thinking:
                    try:
                        # Find the </think> token (ID 151668)
                        index = len(output_ids) - output_ids[::-1].index(151668)
                        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                    except ValueError:
                        # If </think> token not found, treat all as regular content
                        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                else:
                    # If thinking is disabled, all output is regular content
                    content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            if unload_model:
                self.unload_model()
                
            return content, thinking_content
            
        except Exception as e:
            return f"Error during inference: {str(e)}", ""

inferencer = Qwen25Inferencer()

class Qwen2_5Node:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available fp8 types for display
        available_fp8_types = list(SUPPORTED_FP8_TYPES.keys())
        
        # Build torch_dtype options
        torch_dtype_options = ["auto", "float32", "float16", "bfloat16"]
        
        # Add fp8 types
        if available_fp8_types:
            torch_dtype_options.extend(available_fp8_types)
        
        # Build quantization options
        quantization_options = ["auto", "fp16", "bf16", "int8", "int4"]
        
        return {
            "required": {
                "user_prompt": ("STRING", {"default": "Hello, how are you?", "multiline": True}),
                "model_path": ("STRING", {"default": r"你的qwen2或者3模型路径", "multiline": False}),
                "device": (["auto", "cpu", "gpu"], {"default": "auto"}),
                "torch_dtype": (torch_dtype_options, {"default": "auto"}),
                "quantization": (quantization_options, {"default": "auto"}),
                "attn_implementation": (["default", "flash_attention_2", "sdpa", "eager"], {"default": "default"}),
                "max_new_tokens": ("INT", {"default": 32768, "min": 1, "max": 65536, "step": 1}),
                "enable_thinking": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "thinking")
    FUNCTION = "main"
    CATEGORY = "hhy"

    def main(self, user_prompt, model_path, device="auto", torch_dtype="auto", quantization="auto", attn_implementation="default", max_new_tokens=32768, enable_thinking=True, unload_model=False):
        content, thinking_content = inferencer.generate_text(
            user_prompt=user_prompt,
            model_path=model_path,
            device_type=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            unload_model=unload_model,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            quantization=quantization
        )
        return (content, thinking_content)

NODE_CLASS_MAPPINGS = {
    "Qwen2.5": Qwen2_5Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2.5": "Qwen2.5 node"
}