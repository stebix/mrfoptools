import pickle
import logging

from pathlib import Path
from collections.abc import Mapping

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


def store_pickle(
    data: Mapping,
    savepath: Path,
    *,
    overwrite: bool = False
) -> None:
    """Store arbitrary data mapping to pickle file."""
    if savepath.exists() and not overwrite:
        raise FileExistsError(f'Pickle storage failed: file \'{savepath}\' already exists.')
    if not savepath.suffix.endswith('.pkl'):
        savepath = savepath.with_suffix('.pkl')
    with open(savepath, mode='wb') as f:
        pickle.dump(data, f)    
    logger.info(f'Pickle storage successful: data saved to \'{savepath}\'')