
class BatchHelper:

    def __init__(self,  x1, x2, labels, batch_size):
        self.x1 = x1
        # self.x1 = self.x1.reshape(-1, 1)
        self.x2 = x2
        # self.x2 = self.x2.reshape(-1, 1)
        self.labels = labels
        self.labels = self.labels.reshape(-1, 1)
        self.batch_size = batch_size

    def next(self, batch_id):
        x1_batch = self.x1[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
        x2_batch = self.x2[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
        labels_batch = self.labels[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
        return x1_batch, x2_batch, labels_batch