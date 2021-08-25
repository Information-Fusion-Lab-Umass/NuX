# Thanks! https://github.com/google/jax/issues/2116#issuecomment-580322624
import pickle
from pathlib import Path
from typing import Union

suffix = '.pickle'

def save_pytree(data, path: Union[str, Path], overwrite: bool = False):
  path = Path(path)
  if path.suffix != suffix:
    path = path.with_suffix(suffix)
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists():
    if overwrite:
      path.unlink()
    else:
      raise RuntimeError(f'File {path} already exists.')
  with open(path, 'wb') as file:
    pickle.dump(data, file)

def load_pytree(path: Union[str, Path]):
  path = Path(path)
  if not path.is_file():
    raise ValueError(f'Not a file: {path}')
  if path.suffix != suffix:
    raise ValueError(f'Not a {suffix} file: {path}')
  with open(path, 'rb') as file:
    data = pickle.load(file)
  return data
