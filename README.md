# TF_filters_locations-3
Tensorflow for object locators, by using Normalize feature and convolution it creates a new picture with X and Y side shadow of the picture indicate object inside and alignment information about the location. After convert the image it required only one row from the X axis and one row from the Y axis for the extract the object location reflecting to it shadow.

🧸💬 By scanning you need to work on multiple scales of the pictures and multiple sizes of the observing boxes and grouping but working with the alignment shadow is require only the last row from each axis for location information and segmentation of objects in the picture is possible to do it from output array value. </br>

### Pre-process the image input ###

🧸💬 Two layers are used to create a small contrast image from different variances, a smaller mean with a smaller window creates a contrast result and a larger mean with a larger window is create a blur result. </br>
🧸💬 The convolution layer is filled the image with data with the convolution size of the window edge, and create output in image format. </br>
🐑💬 Normalize layer can remove or create the edge of an object in an image with a small contrast value, and remove a lower value by leaving ambingious data. </br>
👧💬 Our eyes work as multiple layers combined by removed of low-dense information the high contrast is information after filtering out noises or dark tone colours. </br>
```
layer_1 = tf.keras.layers.Normalization(mean=3., variance=2.)( image_resized )
layer_2 = tf.keras.layers.Normalization(mean=4., variance=6.)( layer_1 )
image = tf.expand_dims( image_resized, axis=0, name="expand dimension" )
layer_3 = tf.keras.layers.Conv2D(8, (4, 4), activation='relu')( image )
final_layer = tf.keras.layers.Conv2D(1, (1, 1), activation='relu')( layer_3 )
final_layer = tf.squeeze( final_layer, axis=0, name="squeeze" )
image_in_process = final_layer
```
🐬🥀💬 One problem of the convolution image is the channel, we retain output information for each update that guarantees there is no loss of data ( object ) from our work on the screen. </br>
</br>
![alt text](https://github.com/jkaewprateep/TF_filters_locations-3/blob/main/Image_Filters.gif)</br>

### Array counter ###

🧸💬 By mathematics method we do not use search, search sort, or index function we can have a position of the object related to screen resolution by multiplying them with the range array containing the counting number. </br>
```
linespace = tf.linspace(start=1, stop=125, num=125, name=None, axis=0)
linespace = tf.cast( linespace, dtype=tf.float32 )

temp = tf.ones([125, 1])
temp = tf.cast( temp, dtype=tf.float32 )
temp = tf.math.multiply(linespace, temp)
temp = tf.expand_dims(temp, axis=2)
```

### Array counter - output  ###

🐑💬 The sample of the range array we use for the segment location of the object on the screen return from the action of the game library. </br>
```
 [[  1.]
  [  2.]
  [  3.]
  ...
  [123.]
  [124.]
  [125.]]

 [[  1.]
  [  2.]
  [  3.]
  ...
  [123.]
  [124.]
  [125.]]

 [[  1.]
  [  2.]
  [  3.]
  ...
  [123.]
  [124.]
  [125.]]], shape=(125, 125, 1), dtype=float32)
```

### Y-Axis ###

🧸💬 The re-arranging of the object data to one side of the axis, we determine the object's location, shape, and property same as we work with an accurate function. </br>
```
temp2 = tf.argsort(image_in_process, axis=1).numpy() * 1.0
temp2 = temp2 + 1
```

### Y-Axis - output ###

🐑💬 This sample of the output from the new arrangement, object property is collected at one side of the image and we can obtain it with out using the scanning method. </br> 
```
 [[  1.]
  [  2.]
  [125.]
  ...
  [ 67.]
  [ 68.]
  [ 69.]]

 [[  1.]
  [  2.]
  [125.]
  ...
  [ 67.]
  [ 68.]
  [ 69.]]

 [[  1.]
  [  2.]
  [125.]
  ...
  [ 67.]
  [ 68.]
  [ 69.]]], shape=(125, 125, 1), dtype=float32)
```

### X-Axis ###

🧸💬 The re-arranging of the object data to one side of the axis, we determine the object's location, shape, and property same as we work with an accurate function. </br>
```
temp3 = tf.argsort(image_in_process, axis=0).numpy() * 1.0
temp3 = temp3 + 1
```

### X-Axis - output ###

🐑💬 This sample of the output from the new arrangement, object property is collected at one side of the image and we can obtain it with out using the scanning method. </br> 
```
 [[  1.]
  [  2.]
  [125.]
  ...
  [ 67.]
  [ 68.]
  [ 69.]]

 [[  1.]
  [  2.]
  [125.]
  ...
  [ 67.]
  [ 68.]
  [ 69.]]

 [[  1.]
  [  2.]
  [125.]
  ...
  [ 67.]
  [ 68.]
  [ 69.]]], shape=(125, 125, 1), dtype=float32)
```

### Reflecting position in X-Axis ###

🧸💬 We obtain information from the side of the image and we add 1.0 that is because we are planning ahead for the differentiate result we can add-on from these codes. </br> 
```
position_on_x_axis = temp3[123:124,:,0:1]
position_on_x_axis = tf.squeeze( position_on_x_axis, axis=2 )
position_on_x_axis = position_on_x_axis + 1.0

position_on_x_axis = tf.reshape( position_on_x_axis, [125, 1] )
position_on_x_axis = tf.cast( position_on_x_axis, dtype=tf.float32 )
```

### Reflecting position in X-Axis - output  ###

🐑💬 The sample result is extracted from a single line of the input. </br> 
```
tf.Tensor(
[[125.]
 [125.]
 [125.]
 ...
 [125.]
 [ 41.]
 [ 35.]
 [ 34.]
 [ 40.]
 [ 41.]
 [ 42.]
 [ 79.]
 [ 79.]
 [ 79.]
 [ 35.]
 [ 36.]
 [ 76.]
 [ 37.]
 [ 78.]
 [ 76.]
 [125.]
 [125.]
 [125.]
 ...
 [125.]], shape=(125, 1), dtype=float32)
```

### Reflecting position in Y-Axis ###

🧸💬 We obtain information from the side of the image and we add 1.0 that is because we are planning ahead for the differentiate result we can add-on from these codes. </br> 
```
position_on_y_axis = temp3[:,123:124,0:1]
position_on_y_axis = tf.squeeze( position_on_y_axis, axis=2 )
position_on_y_axis = position_on_y_axis + 1.0

position_on_y_axis = tf.reshape( position_on_y_axis, [125, 1] )
position_on_y_axis = tf.cast( position_on_y_axis, dtype=tf.float32 )
```

### Reflecting position in Y-Axis - output  ###

🐑💬 The sample result is extracted from a single line of the input. </br> 
```
tf.Tensor(
[[125.]
 [125.]
 [125.]
 ...
 [125.]
 [ 41.]
 [ 35.]
 [ 34.]
 [ 40.]
 [ 41.]
 [ 42.]
 [ 79.]
 [ 79.]
 [ 79.]
 [ 35.]
 [ 36.]
 [ 76.]
 [ 37.]
 [ 78.]
 [ 76.]
 [125.]
 [125.]
 [125.]
 ...
 [125.]], shape=(125, 1), dtype=float32)
```

### Remove counters and set to 0, image segmentation ###

🧸💬 For custom output ( does not require but clean looking when display as label ) </br> 
```
linespace2 = tf.linspace(start=126, stop=250, num=125, name=None, axis=0)
linespace2 = tf.cast( linespace2, dtype=tf.float32 )
linespace2 = tf.reshape( linespace2, [125, 1] )

position_on_y_axis = tf.where( tf.math.not_equal( linespace2, position_on_y_axis ), linespace2 , 0)
position_on_x_axis = tf.where( tf.math.not_equal( linespace2, position_on_x_axis ), linespace2 , 0)
```

### Remove counters and set to 0 - output, image segmentation ###

🐑💬 The result from the custom output </br> 
```
tf.Tensor(
[[  0.]
 [  0.]
 [  0.]
 ...
 [  0.]
 [144.]
 [145.]
 [146.]
 [147.]
 [148.]
 [149.]
 [150.]
 [151.]
 [152.]  < -- Object 1
 [  0.]
 [  0.]
 [155.]
 [156.]
 [157.]
 [158.]
 [159.]
 [160.]
 [161.]  < -- Object 2
 [  0.]
 [163.]
 [164.]
 [165.]
 [  0.]
 [  0.]
 [168.]
 [169.]
 [170.]
 [171.]  < -- Object 3
 [  0.]
 [  0.]
 [  0.]
 ...
 [  0.]], shape=(125, 1), dtype=float32)
 ```

### Image to display ###

🧸💬 Combined of X-axis and Y-axis images simply summing both values. </br>
```
temp3 = temp3 + temp2
temp3 = tf.cast( temp3, dtype=tf.float32 )
image = tf.keras.utils.array_to_img(temp3)
```

![alt text](https://github.com/jkaewprateep/TF_filters_locations-3/blob/main/SingleObejct_detection.gif)</br>
![alt text](https://github.com/jkaewprateep/TF_filters_locations-3/blob/main/MultipleObject_detection.gif)</br>
![alt text](https://github.com/jkaewprateep/TF_filters_locations-3/blob/main/WaterWorld_GamePlay.gif)</br>
![alt text](https://github.com/jkaewprateep/TF_filters_locations-3/blob/main/Movement_detection.gif)</br>
