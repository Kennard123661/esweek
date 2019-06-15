import os
import tensorflow as tf

from data import get_feature_data
from data import NUM_INPUT_FEATURES
from tqdm import tqdm
import numpy as np

import argparse

train_data, train_labels, val_data, val_labels, \
    test_data, test_labels = get_feature_data()

LABEL_VALUES = [1,2,3,4,5,6,7,8,9,10]



def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='seed for random ops')
    parser.add_argument('--epoch', type=int, default=10000, help='seed for random ops')
    parser.add_argument('--gpu', type=int, default=0, help='seed for random ops')
    args = parser.parse_args()
    return args

def bin_data_and_labels(data, labels):
    # print(data.shape)
    binned_data, binned_labels, binned_ids = list(), list(), list()
    for i in range(len(LABEL_VALUES)):
        label_value = LABEL_VALUES[i]
        binned_idxs = np.argwhere(labels == label_value)
        binned_idxs = [idx[0] for idx in binned_idxs]
        binned_data.append(data[binned_idxs])
        binned_labels.append(labels[binned_idxs])
        binned_ids.append(binned_idxs)
    return binned_data, binned_labels, binned_ids

class Model:
    def __init__(self, sess, learning_rate=3e-4, margin=0.5, regularization_factor=10e-5, batch_size=1000):
        self._sess = sess        
        self._batch_size = batch_size
        self._regularization_factor = regularization_factor
        self._margin = margin
        self._learning_rate = learning_rate

        self._build_model()
        self._setup_train_ops()
        init = tf.global_variables_initializer()
        self._sess.run(init)

    def _build_model(self):
        self._regularization_loss = 0

        self.anchor = tf.placeholder(dtype=tf.float32, shape=[None, NUM_INPUT_FEATURES])
        self.positive = tf.placeholder(dtype=tf.float32, shape=[None, NUM_INPUT_FEATURES])
        self.negative = tf.placeholder(dtype=tf.float32, shape=[None, NUM_INPUT_FEATURES])
        
        self.anchor_labels = tf.placeholder(dtype=tf.int32, shape=[None, 1]) 
        self.positive_labels = tf.placeholder(dtype=tf.int32, shape=[None, 1]) 
        self.negative_labels = tf.placeholder(dtype=tf.int32, shape=[None, 1]) 

        self.anchor_feat, self.anchor_logits = self._siamese_branch(self.anchor)
        self.positive_feat, self.positive_logits = self._siamese_branch(self.positive)
        self.negative_feat, self.negative_logits = self._siamese_branch(self.negative)

        self.anchor_cls = self._get_cls_loss(self.anchor_labels, self.anchor_logits)
        self.positive_cls = self._get_cls_loss(self.positive_labels, self.positive_logits)
        self.negative_cls = self._get_cls_loss(self.negative_labels, self.negative_logits)

        self.cls_loss = (self.anchor_cls + self.positive_cls + self.negative_cls)
        self.triplet_loss = self._get_triplet_loss(\
            self.anchor_feat, self.positive_feat, self.negative_feat)
        # self._regularization_loss = 0 
        self.loss = (self.triplet_loss + self.cls_loss + (self._regularization_loss / 3)) / 3
        # self.loss = self.anchor_cls
        self.predictions = tf.argmax(self.anchor_logits, axis=-1, name='predictions')

    def _get_triplet_loss(self, anchor, positive, negative):
        self.positive_loss = tf.reduce_mean(tf.abs(positive - anchor), axis=-1)
        self.negative_loss = tf.reduce_mean(tf.abs(negative - anchor), axis=-1)
        triplet_loss = tf.maximum(0.0, self.positive_loss - self.negative_loss + self._margin)
        triplet_loss = tf.reduce_mean(triplet_loss)
        return triplet_loss

    def _setup_train_ops(self):
        self.train_op = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.loss)

    def _get_cls_loss(self, labels, logits):
        labels = tf.one_hot(labels - 1, depth=len(LABEL_VALUES))
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

    def _siamese_branch(self, inputs):
        with tf.variable_scope('branch', reuse=tf.AUTO_REUSE):
            out = tf.layers.dense(inputs, units=4, activation=tf.nn.relu, \
                kernel_initializer=tf.initializers.glorot_normal, name='d1')
            self._regularization_loss += tf.nn.l2_loss(out) * self._regularization_factor
            out = tf.layers.dense(inputs, units=8, activation=tf.nn.relu, \
                kernel_initializer=tf.initializers.glorot_normal, name='d2')
            self._regularization_loss += tf.nn.l2_loss(out) * self._regularization_factor          
            feature_out = out
            out = tf.layers.dense(inputs, units=len(LABEL_VALUES), activation=None,
                kernel_initializer=tf.initializers.glorot_normal, name='logits')
            self._regularization_loss += tf.nn.l2_loss(out) * self._regularization_factor
        return feature_out, out

    def train(self, train_anchor, train_pos, train_neg, \
            anchor_labels, positive_labels, negative_labels):
        loss, _ = self._sess.run([self.loss, self.train_op], feed_dict={
            self.anchor: train_anchor,
            self.positive: train_pos,
            self.negative: train_neg,
            self.anchor_labels: anchor_labels,
            self.positive_labels: positive_labels,
            self.negative_labels: negative_labels
        })
        

    def evaluate(self, data, labels):
        predictions = self._sess.run(self.predictions, feed_dict={self.anchor: data})
        # print(min(predictions))
        # print(max(predictions))
        # exit()
        labels = np.reshape(labels - 1, newshape=[-1])
        is_equal = predictions == labels
        accuracy = np.count_nonzero(is_equal) / len(labels)
        print('accuracy is %d', accuracy)

def get_training_triplets(data, labels):
    binned_data, binned_labels, binned_ids = bin_data_and_labels(data, labels)

    shuffle_idxs = np.random.permutation(len(data))
    shuffled_data = data[shuffle_idxs]
    shuffled_labels = labels[shuffle_idxs]
    
    positives, negatives = list(), list()
    positive_labels, negative_labels = shuffled_labels, list()
    for i in range(len(data)):
        anchor = data[i]
        label = labels[i]
        # print(label)
        pos_bin = binned_data[label[0] - 1]
        rand_idx = np.random.randint(len(pos_bin), size=[1])[0]
        
        # positive data
        positives.append(pos_bin[rand_idx])
        
        # negative data
        while True:
            rand_idx = np.random.randint(len(data), size=[1])[0]
            neg_label = labels[rand_idx]
            if neg_label[0] != label[0]:
                negatives.append(data[rand_idx])
                negative_labels.append(neg_label)
                break
    assert len(positives) == len(negatives)
    assert len(data) == len(positives)
    assert len(positive_labels) == len(positives)
    assert len(negative_labels) == len(negatives)
    assert len(data) == len(labels)
    # print(np.array(negatives).shape)
    # print(np.array(positives).shape)
    # print(np.array(shuffled_data).shape)
    # exit()
    return shuffled_data, shuffled_labels, positives, positive_labels, \
        negatives, negative_labels

    

if __name__ == "__main__":
    args = _parse_arguments()

    # set the random seed for repeatability of experimental results
    np.random.seed(seed=args.seed)
    tf.random.set_random_seed(seed=args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    sess = tf.Session()
    model = Model(sess)

    max_epochs = args.epoch
    for epoch in range(max_epochs):
        anchor_data, anchor_labels, positive_data, positive_labels, \
            negative_data, negative_labels = get_training_triplets(train_data, train_labels)
        model.train(anchor_data, positive_data, negative_data, \
            anchor_labels, positive_labels, negative_labels)
        if epoch % 100 == 0:
            print('epoch {0}/{1}'.format(epoch, max_epochs))
            model.evaluate(test_data, test_labels)
    sess.close()
