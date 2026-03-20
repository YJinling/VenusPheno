class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.counter = 0
        self.early_stop = False

    def step(self, metric):
        if self.best is None:
            self.best = metric
            return False

        if metric < self.best + self.delta:
            self.counter += 1
            print(f"[EarlyStopping] no improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best = metric
            self.counter = 0
        return False
