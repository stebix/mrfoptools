import os
from pathlib import Path

import numpy as np
import zarr
import rich
import rich.console


from mrfoptools.io.bag import store_datalike, store_metadatalike


def test_store_datalike(tmp_path):
    test_dir = tmp_path / 'test-1'
    test_dir.mkdir()

    NR = 100
    NSTEPS = 10

    array_path = test_dir / 'array.zarr'
    test_group_name = 'test_group'
    test_group_path = array_path / test_group_name

    test_group = zarr.create_group(test_group_path)
    metadata_group = zarr.create_group(array_path / 'metadata')

    data = {
        'test_array_1' : np.zeros((NSTEPS, NR)),
        'test_array_2' : np.full((NSTEPS, NR), fill_value=25),
        'test_irregular_array' : {
            'sampling_steps' : np.arange(33),
            'sampling_values' : np.full((33, 15), fill_value=1337)
        }
    }

    metadata = {
        'test_metadata_1' : 'test',
        'parameter' : 1000,
        'learning_rate' : 0.01,
        'in_depth_settings' : {
            'n_iterations' : 1000,
            'n_samples' : 100,
            'bathtub_parameters' : {
                'alpha' : 0.1,
                'beta' : 0.3,
                'gamma' : 0.7,
                'delta' : 2.5
            }
        }
    }

    store_datalike(test_group, data)
    store_metadatalike(metadata_group, metadata)

    print(test_group.tree())
    rich.print(metadata_group.attrs.asdict())


class Printer:
    def __init__(
            self,
            console: rich.console.Console | None = None,
            verbose: bool = False,

            ):
        self.verbose = verbose
        self.console = console
        self.print_fn = console.print if console else print

    def print(self, *args, **kwargs):
        if not self.verbose:
            return
        self.print_fn(*args, **kwargs)


def teardown(
        directory: Path,
        console: rich.console.Console | None,
        verbose: bool = False    
    ) -> None:

    printer = Printer(console, verbose)
    if console:
        prefix = '[bold red]'
        suffix = '[/bold red]'
    else:
        prefix = ''
        suffix = ''

    for file in directory.iterdir():
        if file.is_dir():
            printer.print(f'Entering directory: {prefix} {file.resolve()} {suffix}')
            teardown(file, console)
            continue
        printer.print(f'Unlinking: {prefix} {file.resolve()} {suffix}')
        os.unlink(file)

    printer.print(f'Removing directory: {prefix} {directory.resolve()} {suffix}')
    directory.rmdir()    


if __name__ == '__main__':
    tmp_path = Path('/tmp/mrfopt-demo-tmpdir')
    tmp_path.mkdir()

    console = rich.console.Console()

    try:
        test_store_datalike(tmp_path)
    except Exception as e:
        print(f'Error: {e}')
    finally:
        console.rule('Teardown')
        teardown(tmp_path, console)
        console.print('Teardown complete.')
