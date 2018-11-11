import os
import argparse

import numpy as np
from keras import callbacks, optimizers
from learning_rate import create_lr_schedule

from loss import dice_coef_loss, dice_coef, recall, precision
from datasets.coco import DataGenerator as coco_generator
from nets.MobileUNet import MobileUNet

def train(coco_path, checkpoint_path, log_path, epochs=100, batch_size=50):
    cat_nms = ['book', 'apple', 'keyboard']

    BATCH_SIZE = batch_size
    NUM_EPOCH = epochs
    IMAGE_W = 224
    IMAGE_H = 224

    model = MobileUNet(input_shape=(IMAGE_H, IMAGE_W, 3),
                       alpha_up=0.25, num_classes=(len(cat_nms)+1))
    # model.load_weights(os.path.expanduser(mobilenet_weights_path.format(img_height)),
    #            by_name=True)

    # # Freeze mobilenet original weights
    # for layer in model.layers[:82]:
    #     layer.trainable = False
    
    seed = 1
    np.random.seed(seed)

    training_generator = coco_generator(cat_nms, coco_path, batch_size=BATCH_SIZE)
    validation_generator = coco_generator(cat_nms, coco_path, subset='val', batch_size=BATCH_SIZE)

    model.summary()
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path, by_name=True)

    model.compile(
        optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
        # optimizer=Adam(lr=0.001),
        # optimizer=optimizers.RMSprop(),
        loss=dice_coef_loss,
        metrics=[
            dice_coef,
            recall,
            precision,
            'binary_crossentropy',
        ],
    )

    lr_base = 0.01 * (float(BATCH_SIZE) / 16)

    # callbacks
    scheduler = callbacks.LearningRateScheduler(
        create_lr_schedule(NUM_EPOCH, lr_base=lr_base, mode='progressive_drops'))
    tensorboard = callbacks.TensorBoard(log_dir=log_path)
    csv_logger = callbacks.TensorBoard(log_path)
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           save_best_only=True)

    # Train model on dataset
    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        callbacks=[scheduler, tensorboard, checkpoint, csv_logger],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--coco_path',
        type=str,
        default='./data/coco',
        # default='/Volumes/SercanHDD/coco',
        help='data path of coco'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoints/weights.hdf5',
        help='mask file as numpy format'
    )
    parser.add_argument(
        '--log_path',
        type=str,
        default='./logs',
        help='mask file as numpy format'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=250,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
    )
    args, _ = parser.parse_known_args()

    train(**vars(args))