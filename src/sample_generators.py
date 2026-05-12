import jax
from jax import Array
import jax.numpy as jnp


def generate_moons(
        key: Array,
        negative_sample_number: int = 2**20,
        positive_sample_number: int = 2**20,
        feature_noise_std_dev: float = 0.2
        ) -> tuple[Array, Array]:
    sample_number = negative_sample_number + positive_sample_number

    negative_sample_theta = jnp.linspace(0.0, jnp.pi, negative_sample_number)
    negative_sample_x = jnp.cos(negative_sample_theta)
    negative_sample_y = jnp.sin(negative_sample_theta)

    positive_sample_theta = jnp.linspace(0.0, jnp.pi, positive_sample_number)
    positive_sample_x = 1.0 - jnp.cos(positive_sample_theta)
    positive_sample_y = 1.0 - jnp.sin(positive_sample_theta) - 0.5

    feature_collections = jnp.empty(shape=(sample_number, 2)) \
        .at[:negative_sample_number, 0].set(negative_sample_x) \
        .at[:negative_sample_number, 1].set(negative_sample_y) \
        .at[negative_sample_number:, 0].set(positive_sample_x) \
        .at[negative_sample_number:, 1].set(positive_sample_y)

    key, subkey = jax.random.split(key)
    feature_collections = feature_collections \
        + feature_noise_std_dev*jax.random.normal(
            subkey, shape=feature_collections.shape
        )

    labels = jnp.empty(sample_number, dtype=jnp.uint8) \
        .at[:negative_sample_number].set(0) \
        .at[negative_sample_number:].set(1)

    sample_idxs = jnp.arange(sample_number)

    key, subkey = jax.random.split(key)
    shuffled_sample_idxs = jax.random.permutation(subkey, sample_idxs)

    feature_collections = feature_collections[shuffled_sample_idxs]
    labels = labels[shuffled_sample_idxs]

    return feature_collections, labels
