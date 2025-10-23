from collections import deque

class EarlyStopping:
    """Early stopping utility class"""
    def __init__(self, patience:int, window_size:int, min_delta:float, reset_step:float, verbose=False):
        enable = (patience is not None and patience > 0)
        self._enable = enable  # whether to enable early stopping
        if not self._enable:
            print("EarlyStopping is disabled.")
            patience = float('inf')

        self._patience = patience    # maximum allowed counter before stopping
        self._window_size = window_size  # number of epochs to consider for improvement
        self._verbose = verbose      # whether to print logs
        self._counter = 0            # current counter value
        self._min_delta = min_delta  # minimum improvement required
        self._reset_step = int(patience * reset_step)  # decay factor for counter reset

        self._history = deque(maxlen=window_size)   # recent scores history
        
    def __call__(self, score):
        
        # wait until we have enough history
        if len(self._history) + 1 < self._history.maxlen:
            self._history.append(score)
            return
        max_score = max(self._history)
        if score > max_score - self._min_delta:  
            # If the score did not improve, increase counter
            self._counter += 1
            if self._verbose:  
                # Print current counter status
                print(f'EarlyStopping counter: {self._counter} out of {self._patience}')
        else:  
            self._counter = max(0, self.counter - self._reset_step)

        self._history.append(score)  # Update history with the latest score

    @property
    def counter(self):
        return self._counter
    
    @property
    def patience(self):
        return self._patience

    @property
    def early_stop(self):
        if self._counter >= self._patience:  
            # If counter exceeds patience, set stop flag
            return True
        return False

    @staticmethod
    def from_params(params: dict) -> 'EarlyStopping':
        return EarlyStopping(**params)