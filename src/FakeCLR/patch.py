# -------- PATCH training_loop to ignore missing resume keys --------
file_path = 'training/training_loop.py'

with open(file_path, 'r') as f:
    content = f.read()

# Replace strict loading with safe loading
content = content.replace(
    "misc.copy_params_and_buffers(resume_data[name], module, require_all=False)",
    "misc.copy_params_and_buffers(resume_data[name], module, require_all=False) if name in resume_data else None"
)

# ALSO fix unwrap (keeping it safe)
content = content.replace(
    "module = module.module",
    "module = module.module if hasattr(module, 'module') else module"
)

with open(file_path, 'w') as f:
    f.write(content)

print("✅ Patched missing D_ema + unwrap")