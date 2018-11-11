import numpy as np
from keras import callbacks, optimizers
from learning_rate import create_lr_schedule

from loss import dice_coef_loss, dice_coef, recall, precision
from datasets.coco import DataGenerator as coco_generator
from nets.MobileUNet import MobileUNet

if __name__ == '__main__':
    cat_nms = ['book', 'apple', 'keyboard']
    cat_clrs = [[0., 0., 128.], [128., 0., 0.], [0., 128., 0.]]

    BATCH_SIZE = 50
    NUM_EPOCH = 100
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

    training_generator = coco_generator(cat_nms, '/Volumes/SercanHDD/coco', batch_size=BATCH_SIZE)
    validation_generator = coco_generator(cat_nms, '/Volumes/SercanHDD/coco', subset='val', batch_size=BATCH_SIZE)

    model.summary()
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
    tensorboard = callbacks.TensorBoard(log_dir='./logs')
    csv_logger = callbacks.CSVLogger('./logs/training.csv')
    checkpoint = callbacks.ModelCheckpoint(filepath='./checkpoints',
                                           save_weights_only=True,
                                           save_best_only=True)

    # Train model on dataset
    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        callbacks=[scheduler, tensorboard, checkpoint, csv_logger],
    )