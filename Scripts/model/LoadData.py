
class DataLoader(object):
    def __init__(self, data, batch, n_classes, n_way):
        self.data = data
        self.batch = batch
        self.n_way = n_way
        self.n_classes = n_classes

    def get_next_episode(self):
        n_examples = 20
        support_batches = np.zeros([self.batch, self.n_way**2, WIDTH, HEIGHT, 1], dtype=np.float32)
        query_batches = np.zeros([self.batch, self.n_way**2, WIDTH, HEIGHT, 1], dtype=np.float32)
        labels_batches = np.zeros([self.batch, self.n_way**2])
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i_batch in range(self.batch):
            support = np.zeros([self.n_way, WIDTH, HEIGHT, 1])
            query = np.zeros([self.n_way, WIDTH, HEIGHT, 1])
            for i, i_class in enumerate(classes_ep):
                selected = np.random.permutation(n_examples)[:2]
                support[i] = self.data[i_class, selected[0]]
                query[i] = self.data[i_class, selected[1]]
            support_batch = np.take(support, [i // self.n_way for i in
                                                      range(self.n_way ** 2)], axis=0)
            query_batch = tf.tile(query, [self.n_way, 1, 1, 1])

            support_batches[i_batch] = support_batch
            query_batches[i_batch] = query_batch
            labels_batches[i_batch] = [i==j for i in range(self.n_way) for j in range(self.n_way)]

        support_batches = np.vstack(support_batches)
        query_batches = np.vstack(query_batches)
        labels_batches = labels_batches.reshape([-1, 1])

        return support_batches, query_batches, labels_batches