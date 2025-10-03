import jax
import jax.numpy as jnp

from mrfoptools.epg.sequences.jax.fisp_prep import simulate_prepped_fisp



class Test_simulate_prepped_fisp:
    """
    Test the simulate_prepped_fisp function.
    """

    def test_simulate_prepped_fisp(self):
        # Define parameters
        lower_fa: jax.Array = jnp.deg2rad(5.0)
        upper_fa: jax.Array = jnp.deg2rad(90.0)

        ETL: int = 768

        key = jax.random.key(seed=42)

        fa = jax.random.uniform(
            key=key,
            minval=lower_fa,
            maxval=upper_fa,
            shape=ETL
        )

        TR = jnp.full_like(fa, fill_value=12.0)
        phases = jnp.zeros_like(fa)
        preptimes = jnp.array([400, 800])
        T1 = 2500.0
        T2 = 750.0
        M0 = 1.0
        TE = 2.2
        TI = 20.0
        inversion_efficiency = 1.0
        max_states = 1000

        signal = simulate_prepped_fisp(
            fa=fa,
            TR=TR,
            preptimes=preptimes,
            T1=T1,
            T2=T2,
            M0=M0,
            phases=phases,
            TE=TE,
            TI=TI,
            inversion_efficiency=inversion_efficiency,
            max_states=max_states
        )

        print("Signal shape:", signal.shape)
        print("Signal:", signal)