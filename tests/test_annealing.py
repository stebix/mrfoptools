import pytest

import jax
import jax.numpy as jnp


"""


from mrfoptools.optimization.simulated_annealing.annealing_rebuild import (
    Designation, ParameterType, ParameterAxis,
    _sample_perturbation_costs, sample_perturbation_costs,
    generate_randomized_variable, EdgeMode, sweep_sample_perturbations_costs
)




class Test_Designation:

    @pytest.mark.parametrize('type_str', ['fa', 'tr'])
    @pytest.mark.parametrize('axis_str', ['x', 'y'])
    def test_generation_from_valid_string(self, type_str, axis_str):
        designation = Designation.from_strings(type_str, axis_str)
        assert designation.type == ParameterType(type_str)
        assert designation.axis == ParameterAxis(axis_str)

    @pytest.mark.parametrize('type_str', ['fa', 'tr'])
    @pytest.mark.parametrize('axis_str', ['x', 'y'])
    def test_string_representation(self, type_str, axis_str):
        designation = Designation.from_strings(type_str, axis_str)
        assert designation.to_string() == f'{type_str},{axis_str}'



class Test_sample_perturbation_cost_private:

    def test_smoke(self):
        # setup
        seed = 1701
        key = jax.random.key(seed)
        n_controlpoints: int = 7
        perturb_designation = Designation.from_strings('fa', 'x')
        sample_count: int = 250

        def cost_func(arr1, arr2, arr3, arr4):
            return jnp.sum(arr1 + arr2 + arr3 + arr4)

        designations = [
            Designation.from_strings('fa', 'x'),
            Designation.from_strings('fa', 'y'),
            Designation.from_strings('tr', 'y'),
            Designation.from_strings('tr', 'x'),
        ]
        key, *subkeys = jax.random.split(key, num=len(designations)+1)

        variables = [
            generate_randomized_variable(
                key=subkeys[i],
                n_controlpoints=n_controlpoints,
                bounds=(0.0, 500.0) if designation.axis is ParameterAxis.X else (0.0, 1.6),
                relscale=1,
                designation=designation,
                sort=True if designation.axis is ParameterAxis.X else False,
                edge_mode=EdgeMode.FIXED if designation.axis is ParameterAxis.X else EdgeMode.FLOATING,
            )
            for i, designation in enumerate(designations)
        ]

        pcosts = _sample_perturbation_costs(
            key=key,
            designation=perturb_designation,
            variables=variables,
            sample_count=sample_count,
            cost_func=cost_func
        )

        assert pcosts.shape == (sample_count,)





class Test_sample_perturbation_cost:

    def test_smoke(self):
        # setup
        seed = 1701
        key = jax.random.key(seed)
        n_controlpoints: int = 7
        sample_count: int = 250

        def cost_func(arr1, arr2, arr3, arr4):
            return jnp.sum(arr1 + arr2 + arr3 + arr4)

        designations = [
            Designation.from_strings('fa', 'x'),
            Designation.from_strings('fa', 'y'),
            Designation.from_strings('tr', 'y'),
            Designation.from_strings('tr', 'x'),
        ]
        key, *subkeys = jax.random.split(key, num=len(designations)+1)

        variables = [
            generate_randomized_variable(
                key=subkeys[i],
                n_controlpoints=n_controlpoints,
                bounds=(0.0, 500.0) if designation.axis is ParameterAxis.X else (0.0, 1.6),
                relscale=1.0,
                designation=designation,
                sort=True if designation.axis is ParameterAxis.X else False,
                edge_mode=EdgeMode.FIXED if designation.axis is ParameterAxis.X else EdgeMode.FLOATING,
            )
            for i, designation in enumerate(designations)
        ]

        pcosts = sample_perturbation_costs(
            key=key,
            variables=variables,
            sample_count=sample_count,
            cost_func=cost_func,
            pbar_kwargs={'unit' : 'iters'}
        )

        assert len(pcosts) == len(variables)


class Test_find_scale_factors:

    def test_smoke(self):
        # setup
        seed = 1701
        key = jax.random.key(seed)
        j_max: int = 5
        sample_count: int = 250
        n_controlpoints: int = 7

        def cost_func(arr1, arr2, arr3, arr4):
            return jnp.sum(arr1 + arr2 + arr3 + arr4)

        designations = [
            Designation.from_strings('fa', 'x'),
            Designation.from_strings('fa', 'y'),
            Designation.from_strings('tr', 'y'),
            Designation.from_strings('tr', 'x'),
        ]
        key, *subkeys = jax.random.split(key, num=len(designations)+1)
        variables = [
            generate_randomized_variable(
                key=subkeys[i],
                n_controlpoints=n_controlpoints,
                bounds=(0.0, 500.0) if designation.axis is ParameterAxis.X else (0.0, 1.6),
                relscale=1.0,
                designation=designation,
                sort=True if designation.axis is ParameterAxis.X else False,
                edge_mode=EdgeMode.FIXED if designation.axis is ParameterAxis.X else EdgeMode.FLOATING,
            )
            for i, designation in enumerate(designations)
        ]

        sf = sweep_sample_perturbations_costs(key=key, variables=variables, j_max=j_max, sample_count=sample_count, cost_func=cost_func)

        print(sf)

"""