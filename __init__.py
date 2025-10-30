import os
import importlib
import importlib.util
import sys
import jwt
import ctypes
import platform
from pathlib import Path
from datetime import datetime, timezone

current_dir = Path(__file__).parent
DISABLE_AUTH_INJECTION = True
WEB_DIRECTORY = "./web"

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def colored_print(message, color=Colors.WHITE):
    print(f"{color}{message}{Colors.RESET}")

def info_print(message):
    colored_print(message, Colors.CYAN)

def success_print(message):
    colored_print(message, Colors.GREEN)

def warning_print(message):
    colored_print(message, Colors.YELLOW)

def error_print(message):
    colored_print(message, Colors.RED)

class AuthenticationError(Exception):
    pass

class TokenExpiredError(AuthenticationError):
    pass

_SECRET_KEY = None

def load_secret_key():
    """从 keys_config.hhy 加载JWT密钥（仅支持加密的 .hhy 文件）"""
    global _SECRET_KEY
    if _SECRET_KEY is not None:
        return _SECRET_KEY
    try:
        hhy_path = current_dir / "keys_config.hhy"
        if not hhy_path.exists():
            raise FileNotFoundError("密钥配置文件不存在: keys_config.hhy")
        
        # 解密加载
        import base64
        from cryptography.fernet import Fernet
        
        ENCRYPTION_KEY = b'hhy_comfyui_keys_encryption_v1_2025_secret_key_for_obfuscation=='
        key = base64.urlsafe_b64encode(ENCRYPTION_KEY[:32])
        cipher = Fernet(key)
        
        with open(hhy_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_code = cipher.decrypt(encrypted_data).decode('utf-8')
        ns = {}
        exec(decrypted_code, ns)
        
        if 'JWT_SECRET_KEY' not in ns:
            raise AttributeError("密钥配置文件中未找到 JWT_SECRET_KEY")
        _SECRET_KEY = ns['JWT_SECRET_KEY']
        return _SECRET_KEY
    except Exception as e:
        error_print(f"[HHY Nodes] ❌ 加载JWT密钥失败: {e}")
        raise AuthenticationError(f"密钥加载失败: {e}")

def _validate_jwt_format(encrypted_config):
    if isinstance(encrypted_config, list):
        encrypted_config = next((x for x in encrypted_config if isinstance(x, str) and x.strip()), "")
    if not isinstance(encrypted_config, str) or not encrypted_config.strip():
        return None, "请提供有效的认证令牌"
    token = encrypted_config.strip()
    if token.count('.') != 2:
        return None, "认证令牌格式无效，请检查令牌是否完整"
    return token, None

def _decode_jwt_with_secret(token, secret_key):
    try:
        decoded_data = jwt.decode(token, secret_key, algorithms=['HS256'])
        if 'exp' in decoded_data:
            exp_time = datetime.fromtimestamp(decoded_data['exp'], tz=timezone.utc)
            if datetime.now(tz=timezone.utc) > exp_time:
                raise TokenExpiredError("认证令牌已过期，请重新获取")
        return decoded_data
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("认证令牌已过期，请重新获取")
    except jwt.InvalidSignatureError:
        raise AuthenticationError("认证令牌签名无效，请检查令牌来源")
    except jwt.DecodeError:
        raise AuthenticationError("认证令牌格式错误，无法解析")
    except jwt.InvalidTokenError as e:
        error_msg = str(e)
        if "Not enough segments" in error_msg:
            raise AuthenticationError("认证令牌格式无效，请检查令牌是否完整")
        elif "Invalid header" in error_msg:
            raise AuthenticationError("认证令牌头部格式错误")
        elif "Invalid payload" in error_msg:
            raise AuthenticationError("认证令牌内容格式错误")
        else:
            raise AuthenticationError("认证令牌无效，请提供正确的令牌")
    except Exception as e:
        raise AuthenticationError("认证验证失败，请检查令牌")

def require_auth(encrypted_config):
    token, error = _validate_jwt_format(encrypted_config)
    if error:
        raise AuthenticationError(error)
    secret_key = load_secret_key()
    return _decode_jwt_with_secret(token, secret_key)

def create_enhanced_node_module(original_file):
    try:
        module_name = original_file.stem
        with open(original_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        current_dir_path = str(current_dir).replace('\\', '\\\\')
        # 仅支持 .hhy（加密）格式的密钥配置
        keys_config_code = ""
        hhy_path = current_dir / "keys_config.hhy"
        if hhy_path.exists():
            keys_config_code = '''
# 加载密钥配置（加密格式）
import base64 as _b64
from cryptography.fernet import Fernet as _Fernet
_ENCRYPTION_KEY = b'hhy_comfyui_keys_encryption_v1_2025_secret_key_for_obfuscation=='
_key = _b64.urlsafe_b64encode(_ENCRYPTION_KEY[:32])
_cipher = _Fernet(_key)
with open(r"''' + str(hhy_path).replace('\\', '\\\\') + '''", 'rb') as _f:
    _encrypted = _f.read()
_decrypted = _cipher.decrypt(_encrypted).decode('utf-8')
_keys_ns = {}
exec(_decrypted, _keys_ns)
# 创建keys_config模块对象
import types as _types
keys_config = _types.SimpleNamespace(**{k: v for k, v in _keys_ns.items() if not k.startswith('_')})
'''
        else:
            raise FileNotFoundError(f"密钥配置缺失: {hhy_path}")
        
        # 获取原始文件路径
        original_file_path = str(original_file).replace('\\', '\\\\')
        
        enhanced_code = f'''
import sys
import jwt
from pathlib import Path
from datetime import datetime, timezone
{keys_config_code}
CURRENT_DIR_PATH = r"{current_dir_path}"
__file__ = r"{original_file_path}"

def validate_jwt_format(encrypted_config):
    if isinstance(encrypted_config, list):
        encrypted_config = next((x for x in encrypted_config if isinstance(x, str) and x.strip()), "")
    if not isinstance(encrypted_config, str) or not encrypted_config.strip():
        return None, "请提供有效的认证令牌"
    token = encrypted_config.strip()
    if token.count('.') != 2:
        return None, "认证令牌格式无效，请检查令牌是否完整"
    return token, None

def decode_jwt_with_secret(token, secret_key):
    try:
        decoded_data = jwt.decode(token, secret_key, algorithms=['HS256'])
        if 'exp' in decoded_data:
            exp_time = datetime.fromtimestamp(decoded_data['exp'], tz=timezone.utc)
            if datetime.now(tz=timezone.utc) > exp_time:
                raise RuntimeError("认证令牌已过期，请重新获取")
        return decoded_data
    except jwt.ExpiredSignatureError:
        raise RuntimeError("认证令牌已过期，请重新获取")
    except jwt.InvalidSignatureError:
        raise RuntimeError("认证令牌签名无效，请检查令牌来源")
    except jwt.DecodeError:
        raise RuntimeError("认证令牌格式错误，无法解析")
    except jwt.InvalidTokenError as e:
        error_msg = str(e)
        if "Not enough segments" in error_msg:
            raise RuntimeError("认证令牌格式无效，请检查令牌是否完整")
        elif "Invalid header" in error_msg:
            raise RuntimeError("认证令牌头部格式错误")
        elif "Invalid payload" in error_msg:
            raise RuntimeError("认证令牌内容格式错误")
        else:
            raise RuntimeError("认证令牌无效，请提供正确的令牌")
    except Exception as e:
        raise RuntimeError("认证验证失败，请检查令牌")

def require_auth_simple(encrypted_config):
    token, error = validate_jwt_format(encrypted_config)
    if error:
        raise RuntimeError(error)
    # 从keys_config读取JWT密钥
    if 'keys_config' not in globals():
        raise RuntimeError("密钥配置未加载")
    if not hasattr(keys_config, 'JWT_SECRET_KEY'):
        raise RuntimeError("密钥配置中未找到 JWT_SECRET_KEY")
    secret_key = keys_config.JWT_SECRET_KEY
    return decode_jwt_with_secret(token, secret_key)

{original_code}

if 'NODE_CLASS_MAPPINGS' in locals():
    for class_name, node_class in NODE_CLASS_MAPPINGS.items():
        try:
            if hasattr(node_class, 'INPUT_TYPES'):
                original_input_types = node_class.INPUT_TYPES
                def create_new_input_types(orig_func):
                    @classmethod
                    def new_input_types(cls):
                        inputs = orig_func()
                        if 'optional' not in inputs:
                            inputs['optional'] = {{}}
                        inputs['optional']['encrypted_config'] = ("STRING", {{"default": "", "multiline": True}})
                        return inputs
                    return new_input_types
                node_class.INPUT_TYPES = create_new_input_types(original_input_types)
            function_name = getattr(node_class, 'FUNCTION', None)
            if function_name and hasattr(node_class, function_name):
                original_method = getattr(node_class, function_name)
                def create_enhanced_method(orig_method):
                    def enhanced_method(self, *args, **kwargs):
                        encrypted_config = kwargs.get('encrypted_config', '')
                        if not encrypted_config and args:
                            last_arg = args[-1]
                            if isinstance(last_arg, list):
                                last_arg = next((x for x in last_arg if isinstance(x, str) and x.strip()), "")
                            if isinstance(last_arg, str):
                                encrypted_config = last_arg
                                args = args[:-1]
                        if isinstance(encrypted_config, list):
                            encrypted_config = next((x for x in encrypted_config if isinstance(x, str) and x.strip()), "")
                        if not {str(DISABLE_AUTH_INJECTION)}:
                            jwt_data = require_auth_simple(encrypted_config)
                            setattr(self, '_jwt_user_data', jwt_data)
                        if 'encrypted_config' in kwargs:
                            del kwargs['encrypted_config']
                        return orig_method(self, *args, **kwargs)
                    return enhanced_method
                setattr(node_class, function_name, create_enhanced_method(original_method))
        except Exception as e:
            pass
'''
        enhanced_namespace = {}
        exec(enhanced_code, enhanced_namespace)
        enhanced_module = type(sys)('enhanced_' + module_name)
        for key, value in enhanced_namespace.items():
            if not key.startswith('__'):
                setattr(enhanced_module, key, value)
        return enhanced_module, None
    except Exception as e:
        return None, str(e)

def discover_and_import_nodes():
    info_print(f"[HHY Nodes] Starting node discovery with memory-only injection in: {current_dir}")
    
    # 检查密钥配置文件（仅支持 .hhy）
    hhy_path = current_dir / "keys_config.hhy"
    if hhy_path.exists():
        success_print("[HHY Nodes] ✓ 找到密钥配置文件: keys_config.hhy")
    else:
        error_print("[HHY Nodes] ✗ 未找到密钥配置文件: keys_config.hhy")
    
    python_files = [f for f in os.listdir(current_dir) 
                   if f.endswith('.py') and f != '__init__.py' and not f.startswith('__')]
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    successful_imports = 0
    failed_imports = []
    for py_file in python_files:
        module_name = py_file[:-3]
        module_file = current_dir / py_file
        try:
            enhanced_module, error = create_enhanced_node_module(module_file)
            if enhanced_module is None:
                failed_imports.append((module_name, error))
                continue
            if hasattr(enhanced_module, 'NODE_CLASS_MAPPINGS'):
                for key, value in enhanced_module.NODE_CLASS_MAPPINGS.items():
                    NODE_CLASS_MAPPINGS[key] = value
                    successful_imports += 1
            if hasattr(enhanced_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                for key, value in enhanced_module.NODE_DISPLAY_NAME_MAPPINGS.items():
                    NODE_DISPLAY_NAME_MAPPINGS[key] = value
            if not hasattr(enhanced_module, 'NODE_CLASS_MAPPINGS') or not enhanced_module.NODE_CLASS_MAPPINGS:
                info_print(f"[HHY Nodes] - Skipped {module_name} (no ComfyUI nodes found)")
        except Exception as e:
            failed_imports.append((module_name, str(e)))
            error_print(f"[HHY Nodes] ✗ Failed to load {module_name}: {str(e)}")
    colored_print(f"[HHY Nodes] Node discovery completed:", Colors.BOLD)
    success_print(f"[HHY Nodes] - Successfully loaded: {successful_imports} modules")
    success_print(f"[HHY Nodes] - Total nodes registered: {len(NODE_CLASS_MAPPINGS)}")
    if failed_imports:
        error_print(f"[HHY Nodes] - Failed imports: {len(failed_imports)}")
        for module_name, error in failed_imports:
            error_print(f"[HHY Nodes]   - {module_name}: {error}")
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = discover_and_import_nodes()
except Exception as e:
    error_print(f"[HHY Nodes] ❌ Critical error during node registration: {str(e)}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]