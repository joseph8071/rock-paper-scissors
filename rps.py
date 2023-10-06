import tensorflow as tf
import tensorflow_datasets as tfds


(train_data, test_data), ds_info = tfds.load(
    'rock_paper_scissors',
    split=['train', 'test'],
    as_supervised=True,  # returns tuple (img, label) instead of dict
    with_info=True  # returns info about the dataset
)

def preprocess_img(image, label):
    # Normalize images to [0,1]
    image = tf.image.resize(image, [150, 150])
    return tf.cast(image, tf.float32) / 255., label

train_data = train_data.map(preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train_data.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = test_data.batch(32)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(train_data, epochs=10, validation_data=test_data)

test_loss, test_accuracy = model.evaluate(test_data)
print("Test accuracy: ", test_accuracy)

model.save('rps_model.h5')