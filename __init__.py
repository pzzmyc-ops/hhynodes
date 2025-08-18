import os
import importlib
import importlib.util
import sys
from pathlib import Path

current_dir = Path(__file__).parent

WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def discover_and_import_nodes():
    print(f"[HHY Nodes] Starting node discovery in: {current_dir}")
    python_files = [f for f in os.listdir(current_dir) 
                   if f.endswith('.py') and f != '__init__.py' and not f.startswith('__')]
    successful_imports = 0
    failed_imports = []
    for py_file in python_files:
        module_name = py_file[:-3]
        try:
            module_path = f"custom_nodes.hhy_nodes.{module_name}"
            spec = importlib.util.spec_from_file_location(module_name, current_dir / py_file)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path] = module
            spec.loader.exec_module(module)
            if hasattr(module, 'NODE_CLASS_MAPPINGS') and module.NODE_CLASS_MAPPINGS:
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                successful_imports += 1
                if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS') and module.NODE_DISPLAY_NAME_MAPPINGS:
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                print(f"[HHY Nodes] ✓ Successfully loaded {module_name} with {len(module.NODE_CLASS_MAPPINGS)} node(s)")
                for node_key, node_class in module.NODE_CLASS_MAPPINGS.items():
                    category = getattr(node_class, 'CATEGORY', 'Unknown')
                    print(f"[HHY Nodes]   - {node_key} (Category: {category})")
            else:
                print(f"[HHY Nodes] - Skipped {module_name} (no ComfyUI nodes found)")
        except Exception as e:
            failed_imports.append((module_name, str(e)))
            print(f"[HHY Nodes] ✗ Failed to load {module_name}: {str(e)}")
    print(f"[HHY Nodes] Node discovery completed:")
    print(f"[HHY Nodes] - Successfully loaded: {successful_imports} modules")
    print(f"[HHY Nodes] - Total nodes registered: {len(NODE_CLASS_MAPPINGS)}")
    if failed_imports:
        print(f"[HHY Nodes] - Failed imports: {len(failed_imports)}")
        for module_name, error in failed_imports:
            print(f"[HHY Nodes]   - {module_name}: {error}")
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = discover_and_import_nodes()
except Exception as e:
    print(f"[HHY Nodes] ❌ Critical error during node registration: {str(e)}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
