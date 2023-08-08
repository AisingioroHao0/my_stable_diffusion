from email.mime import image
from typing import Tuple
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras


class AttentionBlock(keras.layers.Layer):
    """
        采用一乘一卷积的注意力机制,每个位置会映射出filters个查询与结果
        输入输出形状相同，有多少个通道，设置多少个注意力头

    Args:
        filters (int): 相当于注意力头数，结果通道数

    """

    # def __init__(self, filters):

    #     super().__init__()
    #     self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
    #     self.q = PaddedConv2D(filters=filters,kernel_size=1)
    #     self.k = PaddedConv2D(filters=filters,kernel_size=1)
    #     self.v = PaddedConv2D(filters=filters,kernel_size=1)
    #     self.proj_out = PaddedConv2D(filters=filters,kernel_size=1)
    def __init__(self, filters):
        super().__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.q = keras.layers.Conv2D(filters=filters, kernel_size=1)
        self.k = keras.layers.Conv2D(filters=filters, kernel_size=1)
        self.v = keras.layers.Conv2D(filters=filters, kernel_size=1)
        self.proj_out = keras.layers.Conv2D(filters=filters, kernel_size=1)

    def call(self, inputs):
        a = self.norm(inputs)
        q, k, v = self.q(a), self.k(a), self.v(a)

        batch, height, width, channels = q.shape
        # (h*w,c) h*w个位置的查询向量
        q = tf.reshape(q, (-1, height * width, channels))
        k = keras.layers.Permute((3, 1, 2))(k)  # (c,h,w)
        k = tf.reshape(k, (-1, channels, height * width))  # (c,h*w)
        # (h*w,h*w) w[i][j]=q[i][...] dot k[...][j]  行向量有意义  w[i][...]为第i个位置对其他位置查询的权重向量，
        w = q @ k
        w = w * (channels ** (-0.5))
        w = keras.activations.softmax(w)

        v = keras.layers.Permute((3, 1, 2))(v)  # (c,h,w)
        # (c,h*w) v[...][j]为第j个位置的价值向量
        v = tf.reshape(v, (-1, channels, height * width))
        w = keras.layers.Permute((2, 1))(w)  # w[...][j]为第j个位置对其他位置的查询权重
        # (c,h*w) a[i]为第i个注意力头的结果，a[i][j]=v[i][...] dot w[...][j] 第i个注意力头的价值和对应位置的权重相乘求和
        a = v @ w
        a = keras.layers.Permute((2, 1))(a)  # (h*w,c)
        a = tf.reshape(a, (-1, height, width, channels))  # (h,w,c)展开
        return inputs + self.proj_out(a)


class ResConvBlock(keras.layers.Layer):
    """
    swish加工，same padding卷积两次，如果输入输出通道不同，则会用卷积同步输入通道为输出通道，构建残差连接

    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.conv1 = keras.layers.Conv2D(
            filters=out_channels, kernel_size=3, padding="same"
        )
        self.norm2 = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.conv2 = keras.layers.Conv2D(
            filters=out_channels, kernel_size=3, padding="same"
        )
        self.nin_shortcut = (
            keras.layers.Conv2D(filters=out_channels, kernel_size=1, padding="same")
            if in_channels != out_channels
            else lambda x: x
        )

    def call(self, inputs):
        h = self.conv1(keras.activations.swish(self.norm1(inputs)))
        h = self.conv2(keras.activations.swish(self.norm2(h)))
        return self.nin_shortcut(inputs) + h


class ImageEncoder(keras.Sequential):
    """
    使用ResConvBlock进行特征提取，再通过步长为2的卷积减小图像尺寸，通道从128增加到512
    图像尺寸缩小为原来的1/8
    在512通道下使用注意力机制提取特征
    使用swish加工输出，然后再用卷积降到8个通道

    """

    def __init__(self):
        super().__init__(
            [
                # 输入为h,w,c
                keras.layers.Conv2D(filters=128, kernel_size=3, padding="same"),
                ResConvBlock(in_channels=128, out_channels=128),
                ResConvBlock(in_channels=128, out_channels=128),
                keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))),
                keras.layers.Conv2D(filters=128, kernel_size=3, strides=2),
                # h/2,w/2,128
                ResConvBlock(in_channels=128, out_channels=256),
                ResConvBlock(in_channels=256, out_channels=256),
                keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))),
                keras.layers.Conv2D(filters=256, kernel_size=3, strides=2),
                # h/4,w/4,256
                ResConvBlock(in_channels=256, out_channels=512),
                ResConvBlock(in_channels=512, out_channels=512),
                keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))),
                keras.layers.Conv2D(filters=512, kernel_size=3, strides=2),
                # h/8,w/8,512
                ResConvBlock(in_channels=512, out_channels=512),
                ResConvBlock(in_channels=512, out_channels=512),
                ResConvBlock(in_channels=512, out_channels=512),
                AttentionBlock(filters=512),
                ResConvBlock(in_channels=512, out_channels=512),
                tfa.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                keras.layers.Conv2D(filters=8, kernel_size=3, padding="same"),
                keras.layers.Conv2D(filters=8, kernel_size=1),
                keras.layers.Lambda(lambda x: x[..., :4] * 0.18215),
            ]
        )

    @staticmethod
    def get_models(img_height, img_width, download_weights=True)->Tuple[keras.Model,keras.Model]:
        img_latent_height = img_height // 8
        img_latent_width = img_width // 8
        inp_img = keras.layers.Input((img_height, img_width, 3))
        image_encoder = ImageEncoder()
        image_encoder = keras.models.Model(inp_img, image_encoder(inp_img))

        latent = keras.layers.Input((img_latent_height, img_latent_width, 4))
        image_decoder = ImageDecoder()
        image_decoder = keras.models.Model(latent, image_decoder(latent))
        
        image_encoder.summary(line_length=150,expand_nested=True)
        image_decoder.summary(line_length=150,expand_nested=True)
        if download_weights:
            image_decoder_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
                file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
            )
            image_decoder.load_weights(image_decoder_weights_fpath)
            image_encoder_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/divamgupta/stable-diffusion-tensorflow/resolve/main/encoder_newW.h5",
                file_hash="56a2578423c640746c5e90c0a789b9b11481f47497f817e65b44a1a5538af754",
            )
            image_encoder.load_weights(image_encoder_weights_fpath)

        return image_encoder, image_decoder


class ImageDecoder(keras.Sequential):
    """
        输入为8个通道，先一乘一卷积压缩到4个通道，扩展到512通道，再512通道下使用注意力编码，残差卷积，上采样扩大图像尺寸
        使用残差卷积减小通道数量，512到256到128，使用swish加工数据，卷积到三个通道

    Args:
        keras ([type]): [description]
    """

    def __init__(self):
        super().__init__(
            [
                keras.layers.Lambda(lambda x: 1 / 0.18215 * x),
                keras.layers.Conv2D(filters=4, kernel_size=1),
                keras.layers.Conv2D(filters=512, kernel_size=3, padding="same"),
                ResConvBlock(512, 512),
                AttentionBlock(512),
                ResConvBlock(512, 512),
                ResConvBlock(512, 512),
                ResConvBlock(512, 512),
                ResConvBlock(512, 512),
                keras.layers.UpSampling2D(size=(2, 2)),
                keras.layers.Conv2D(filters=512, kernel_size=3, padding="same"),
                ResConvBlock(512, 512),
                ResConvBlock(512, 512),
                ResConvBlock(512, 512),
                keras.layers.UpSampling2D(size=(2, 2)),
                keras.layers.Conv2D(filters=512, kernel_size=3, padding="same"),
                ResConvBlock(512, 256),
                ResConvBlock(256, 256),
                ResConvBlock(256, 256),
                keras.layers.UpSampling2D(size=(2, 2)),
                keras.layers.Conv2D(filters=256, kernel_size=3, padding="same"),
                ResConvBlock(256, 128),
                ResConvBlock(128, 128),
                ResConvBlock(128, 128),
                tfa.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                keras.layers.Conv2D(filters=3, kernel_size=3, padding="same"),
            ]
        )