import ray

import tensorflow as tf
import tensorflow_probability as tfp

#from gnn_uq.gnn_model import RegressionUQSpace, nll
from gnn_uq.load_data import load_data

from autogllight.utils import *
from autogllight.nas.space import (
    SinglePathNodeClassificationSpace,
)
from autogllight.nas.algorithm import (
    RandomSearch,
)
from autogllight.nas.estimator import OneShotEstimator, TrainScratchEstimator


tfd = tfp.distributions
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
DistributionLambda = tfp.layers.DistributionLambda

available_gpus = tf.config.list_physical_devices("GPU")
n_gpus = len(available_gpus)

is_gpu_available = n_gpus > 0

if is_gpu_available:
    print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
else:
    print("No GPU available")

if not (ray.is_initialized()):
    if is_gpu_available:
        ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)
    else:
        ray.init(num_cpus=2, log_to_driver=False)


def trainer(model, dataset, infer_mask, evaluation, *args, **kwargs):
    from autogllight.utils.backend.op import bk_mask, bk_label
    
    # Train
    mask = "train"
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, weight_decay=5e-7)
    device = next(iter(model.trainable_variables)).device
    dset = dataset[0]
    mask = bk_mask(dset, mask)
    epochs = 100
    
    for e in range(epochs):
        with tf.device(device):  # Ensure model is on the correct device
            with tf.GradientTape() as tape:
                pred = model(dset, *args, **kwargs)[mask]
                label = bk_label(dset)
                y = label[mask]
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, pred))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Infer
    mask = infer_mask
    dset = dataset[0]
    mask = bk_mask(dset, mask)
    pred = model(dset, *args, **kwargs)[mask]
    label = bk_label(dset)
    y = label[mask]
    probs = tf.nn.softmax(pred, axis=1).numpy()
    y = y.numpy()
    
    metrics = {}
    for eva in evaluation:
        eval_name = eva.get_eval_name()
        eval_result = eva.evaluate(probs, y)
        metrics[eval_name] = eval_result
    
    print(str(metrics))
    return metrics, loss.numpy()

for seed in range(8):
    datasets = ['delaney', 'freesolv', 'lipo', 'qm7']

    for i, dataset in enumerate(datasets):
        if dataset != 'qm7':
            bs = 50
        else:
            bs = 200

        dataset_load = load_data(dataset, 'random')
        data = dataset[0]
        label = data.y
        input_dim = data.x.shape[-1]
        num_classes = len(np.unique(label.numpy()))

        space = SinglePathNodeClassificationSpace(
            input_dim=input_dim, output_dim=num_classes
        )
        space.instantiate()
        algo = RandomSearch(num_epochs = 100)

        estimator = TrainScratchEstimator(trainer)
    
        ans = algo.search(space, dataset_load, estimator)

        model = ans
        model.save("bb_model.syv")