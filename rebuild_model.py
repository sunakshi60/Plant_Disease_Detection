import tensorflow as tf

def create_model(input_shape=(128, 128, 3), num_classes=38):
    model = tf.keras.models.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        
        # Fourth Convolutional Block
        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        
        # Fifth Convolutional Block
        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1500, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Create the model
    model = create_model()
    
    # Print model summary
    model.summary()
    
    # Save the model
    model.save('trained_plant_disease_model.keras', save_format='keras_v3') 