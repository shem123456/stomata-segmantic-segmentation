
from keras.models import *
from keras.layers import *
import keras.backend as K
import keras

IMAGE_ORDERING = 'channels_last'

def relu6(x):
	return K.relu(x, max_value=6)

alpha = 1
def relu6(x):
    # relu函数
    return K.relu(x, max_value=6.0)

def hard_swish(x):
    # 利用relu函数乘上x模拟sigmoid
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def return_activation(x, nl):
    # 用于判断使用哪个激活函数
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)

    return x

def squeeze(inputs):
    # 注意力机制单元
    input_channels = int(inputs.shape[-1])
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(input_channels/4))(x)
    x = Activation(relu6)(x)
    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):

	channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
	filters = int(filters * alpha)
	x = ZeroPadding2D(padding=(1, 1), name='conv1_pad', data_format=IMAGE_ORDERING  )(inputs)
	x = Conv2D(filters, kernel , data_format=IMAGE_ORDERING  ,
										padding='valid',
										use_bias=False,
										strides=strides,
										name='conv1')(x)
	x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)

	return Activation(relu6, name='conv1_relu')(x)




def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
													depth_multiplier=1, strides=(1, 1), block_id=1):

	channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
	pointwise_conv_filters = int(pointwise_conv_filters * alpha)

	x = ZeroPadding2D((1, 1) , data_format=IMAGE_ORDERING , name='conv_pad_%d' % block_id)(inputs)
	x = DepthwiseConv2D((3, 3) , data_format=IMAGE_ORDERING ,
														 padding='valid',
														 depth_multiplier=depth_multiplier,
														 strides=strides,
														 use_bias=False,
														 name='conv_dw_%d' % block_id)(x)
	x = BatchNormalization(
			axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
	x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

	x = Conv2D(pointwise_conv_filters, (1, 1), data_format=IMAGE_ORDERING ,
										padding='same',
										use_bias=False,
										strides=(1, 1),
										name='conv_pw_%d' % block_id)(x)
	x = BatchNormalization(axis=channel_axis,
																name='conv_pw_%d_bn' % block_id)(x)
	return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)





def get_mobilenet_encoder( input_height=224 ,  input_width=224 , pretrained='imagenet' ):

	alpha=1.0
	depth_multiplier=1
	dropout=1e-3


	img_input = Input(shape=(input_height,input_width , 3 ))


	x = _conv_block(img_input, 32, alpha, strides=(2, 2))
	x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
	# x = squeeze(x)
	f1 = x

	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
														strides=(2, 2), block_id=2)  
	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
	# 注意力模块
	x = squeeze(x)
	f2 = x

	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
														strides=(2, 2), block_id=4)  
	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5) 
	# 注意力模块
	x = squeeze(x)
	f3 = x

	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
														strides=(2, 2), block_id=6) 
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7) 
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8) 
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9) 
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10) 
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
	# x = squeeze(x)
	f4 = x 

	x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
														strides=(2, 2), block_id=12)  
	x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13) 
	# x = squeeze(x)
	f5 = x 

	return img_input , [f1 , f2 , f3 , f4 , f5 ]

