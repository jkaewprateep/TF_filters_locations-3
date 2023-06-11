# TF_filters_locations-3
Tensorflow for object locators

### Pro-process the image input ###

```
layer_1 = tf.keras.layers.Normalization(mean=3., variance=2.)( image_resized )
layer_2 = tf.keras.layers.Normalization(mean=4., variance=6.)( layer_1 )
image = tf.expand_dims( image_resized, axis=0, name="expand dimension" )
layer_3 = tf.keras.layers.Conv2D(8, (4, 4), activation='relu')( image )
final_layer = tf.keras.layers.Conv2D(1, (1, 1), activation='relu')( layer_3 )
final_layer = tf.squeeze( final_layer, axis=0, name="squeeze" )
image_in_process = final_layer
```

### Array counter ###

```
linespace = tf.linspace(start=1, stop=125, num=125, name=None, axis=0)
linespace = tf.cast( linespace, dtype=tf.float32 )

temp = tf.ones([125, 1])
temp = tf.cast( temp, dtype=tf.float32 )
temp = tf.math.multiply(linespace, temp)
temp = tf.expand_dims(temp, axis=2)
```

### Y-Axis ###

```
temp2 = tf.argsort(image_in_process, axis=1).numpy() * 1.0
temp2 = temp2 + 1
```

### X-Axis ###

```
temp3 = tf.argsort(image_in_process, axis=0).numpy() * 1.0
temp3 = temp3 + 1
```

### Reflecting position in X-Axis ###

```
position_on_x_axis = temp3[123:124,:,0:1]
position_on_x_axis = tf.squeeze( position_on_x_axis, axis=2 )
position_on_x_axis = position_on_x_axis + 1.0

position_on_x_axis = tf.reshape( position_on_x_axis, [125, 1] )
position_on_x_axis = tf.cast( position_on_x_axis, dtype=tf.float32 )
```

### Reflecting position in Y-Axis ###

```
position_on_y_axis = temp3[:,123:124,0:1]
position_on_y_axis = tf.squeeze( position_on_y_axis, axis=2 )
position_on_y_axis = position_on_y_axis + 1.0

position_on_y_axis = tf.reshape( position_on_y_axis, [125, 1] )
position_on_y_axis = tf.cast( position_on_y_axis, dtype=tf.float32 )
```

### Remove counters and set to 0 ###

```
linespace2 = tf.linspace(start=126, stop=250, num=125, name=None, axis=0)
linespace2 = tf.cast( linespace2, dtype=tf.float32 )
linespace2 = tf.reshape( linespace2, [125, 1] )

position_on_y_axis = tf.where( tf.math.not_equal( linespace2, position_on_y_axis ), linespace2 , 0)
position_on_x_axis = tf.where( tf.math.not_equal( linespace2, position_on_x_axis ), linespace2 , 0)
```

### Image to display ###

```
temp3 = temp3 + temp2
temp3 = tf.cast( temp3, dtype=tf.float32 )
image = tf.keras.utils.array_to_img(temp3)
```

![alt text](https://github.com/jkaewprateep/TF_filters_locations-3/blob/main/SingleObejct_detection.gif)</br>
![alt text](https://github.com/jkaewprateep/TF_filters_locations-3/blob/main/MultipleObject_detection.gif)</br>
![alt text](https://github.com/jkaewprateep/TF_filters_locations-3/blob/main/WaterWorld_GamePlay.gif)</br>
![alt text](https://github.com/jkaewprateep/TF_filters_locations-3/blob/main/Movement_detection.gif)</br>
