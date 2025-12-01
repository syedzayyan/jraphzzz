import jax.numpy as jnp
import jax.random as jr

### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / jnp.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, key: jr.PRNGKey):
        self.src_list = jnp.unique(src_list)
        self.dst_list = jnp.unique(dst_list)
        self.key = key

    def sample(self, size):
        src_index = jr.randint(minval = 0, maxval = len(self.src_list), shape = size, key=self.key)
        dst_index = jr.randint(minval = 0, maxval = len(self.dst_list), shape = size, key=self.key)
        return self.src_list[src_index], self.dst_list[dst_index]