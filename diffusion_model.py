import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from layers import apply_seq, GEGLU


class ResBlock(keras.layers.Layer):
    """
    融合时间编码的残差卷积
    输入为图像和时间编码，输出为图像和时间编码融合的结果，尺寸不变
    图像使用卷积映射到给定通道数上
    时间编码使用Dense映射到给定的通道数上
    使用卷积变换到给定通道数上，使用Dense编码为每个通道编码，并广播到每个位置上，并附加残差连接
    """

    def __init__(
        self,
        channels,
        out_channels,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs
        )
        self.in_layers = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            keras.layers.Conv2D(filters=out_channels, kernel_size=3, padding="same"),
        ]
        self.emb_layers = [
            keras.activations.swish,
            keras.layers.Dense(units=out_channels),
        ]
        self.out_layers = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            keras.layers.Conv2D(filters=out_channels, kernel_size=3, padding="same"),
        ]
        self.skip_connection = (
            keras.layers.Conv2D(filters=out_channels, kernel_size=1)
            if channels != out_channels
            else lambda x: x
        )

    def call(self, inputs):
        x, emb = inputs
        h = apply_seq(x, self.in_layers)  # (b,h,w,out_channels)
        emb_out = apply_seq(emb, self.emb_layers)  # (b,out_channels)
        h = h + emb_out[:, None, None]
        h = apply_seq(h, self.out_layers)
        ret = self.skip_connection(x) + h
        return ret


class CrossAttention(keras.layers.Layer):
    """
    输入为(h*w,n_heads * d_head)
    输出为
    使用Dense映射到n_heads * d_head维度内
    可以附带context，计算kv，实现的注意力机制，若不附带，则为自注意力机制
    通过reshape，每个头的注意力值将会被压缩到一个轴内
    """

    def __init__(self, n_heads, d_head):
        super().__init__()
        self.to_q = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_k = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_v = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.scale = d_head**-0.5
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = keras.layers.Dense(n_heads * d_head)

    def call(self, inputs):
        assert type(inputs) is list
        if len(inputs) == 1:
            inputs = inputs + [None]
        x, context = inputs
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        assert len(x.shape) == 3
        q = tf.reshape(q, (-1, x.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
        v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

        # (bs, num_heads, time, head_size)
        q = keras.layers.Permute((2, 1, 3))(q)
        # (bs, num_heads, head_size, time)
        k = keras.layers.Permute((2, 3, 1))(k)
        # (bs, num_heads, time, head_size)
        v = keras.layers.Permute((2, 1, 3))(v)

        # score = td_dot(q, k) * self.scale
        score = q @ k * self.scale  # (bs,num_heads,time,time)
        # (bs, num_heads, time, time) 第i行 为其他位置对i位置回应的权重
        weights = keras.activations.softmax(score)
        # attention = td_dot(weights, v)
        attention = weights @ v  # (bs,num_heads,time,head_size)
        attention = keras.layers.Permute((2, 1, 3))(
            attention
        )  # (bs, time, num_heads, head_size)
        h_ = tf.reshape(attention, (-1, x.shape[1], self.num_heads * self.head_size))
        return self.to_out(h_)


class BasicTransformerBlock(keras.layers.Layer):
    """
    输入包含输入，上下文两个部分
    首先使用自注意力编码，然后再使用包含上下文的交叉注意力编码，最终使用dense层映射与给定的dim维度内
    """

    def __init__(self, dim, n_heads, d_head):
        super().__init__()
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(n_heads, d_head)

        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(n_heads, d_head)

        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        x, context = inputs
        x = self.attn1([self.norm1(x)]) + x
        x = self.attn2([self.norm2(x), context]) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class SpatialTransformer(keras.layers.Layer):
    """
    输入为图像，与上下文信息 (b,h,w,c,2)

    使用注意力机制编码，使用1*1卷积将图像映射到到注意力所需要的通道数上n_heads * d_head
    然后reshape到二维(h*w,n_heads * d_head)传入BasicTransformerBlock

    输出为(h,w,channels)
    """

    def __init__(self, channels, n_heads, d_head):
        super().__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
        assert channels == n_heads * d_head
        self.proj_in = keras.layers.Conv2D(filters=n_heads * d_head, kernel_size=1)
        self.transformer_blocks = [
            BasicTransformerBlock(dim=channels, n_heads=n_heads, d_head=d_head)
        ]
        self.proj_out = keras.layers.Conv2D(filters=channels, kernel_size=1)

    def call(self, inputs):
        x, context = inputs
        b, h, w, c = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)  # b,h,w,n_heads * d_head
        x = tf.reshape(x, (-1, h * w, c))
        for block in self.transformer_blocks:
            x = block([x, context])
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + x_in


class DownSample(keras.layers.Layer):
    """
    下采样为原来的二分之一，并通过卷积映射到给定通道数内
    (n-1)/2 +1
    5
    """

    def __init__(self, channels):
        super().__init__()
        self.padding = keras.layers.ZeroPadding2D(padding=(1, 1))
        self.conv2d = keras.layers.Conv2D(filters=channels, kernel_size=3, strides=2)

    def call(self, x):
        x = self.padding(x)
        x = self.conv2d(x)
        return x


class UpSample(keras.layers.Layer):
    """
    上采样为原先尺寸的两倍，并通过卷积映射到给定通道内
    """

    def __init__(self, channels):
        super().__init__()
        self.ups = keras.layers.UpSampling2D(size=(2, 2))
        self.conv = keras.layers.Conv2D(filters=channels, kernel_size=3, padding="same")

    def call(self, x):
        x = self.ups(x)
        return self.conv(x)


class DiffusionModel(keras.models.Model):
    """
        输入包含图像，时间编码，上下文
        time_embed进一步编码时间，使用两个1280units的Dense层，swish
        时间编码使用两个1280个units的Dense层，使用ResBlock会附加时间编码信息，SpatialTransformer则是附加上下文信息

        encoder部分
        输入刚开始通过卷积映射到320通道中
        使用DownSample缩小图像尺寸，ResBlock增加通道，直到1280通道

        decoder部分
        concat 对应encoder层的输出，需要先使用ResBlock恢复通道数，再使用UpSample扩大图像尺寸

    Args:
        keras (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.time_embed = [
            keras.layers.Dense(units=1280),
            keras.activations.swish,
            keras.layers.Dense(units=1280),
        ]
        self.input_blocks = [
            [
                keras.layers.Conv2D(filters=320, kernel_size=3, padding="same")
            ],  # (h,w,320)
            [
                ResBlock(channels=320, out_channels=320),
                SpatialTransformer(channels=320, n_heads=8, d_head=40),
            ],
            [
                ResBlock(channels=320, out_channels=320),
                SpatialTransformer(channels=320, n_heads=8, d_head=40),
            ],
            [DownSample(channels=320)],  # (h/2,w/2.320)
            [
                ResBlock(channels=320, out_channels=640),
                SpatialTransformer(channels=640, n_heads=8, d_head=80),
            ],
            [
                ResBlock(channels=640, out_channels=640),
                SpatialTransformer(channels=640, n_heads=8, d_head=80),
            ],
            [DownSample(channels=640)],  # (h/4,w/2.640)
            [
                ResBlock(channels=640, out_channels=1280),
                SpatialTransformer(channels=1280, n_heads=8, d_head=160),
            ],
            [
                ResBlock(channels=1280, out_channels=1280),
                SpatialTransformer(channels=1280, n_heads=8, d_head=160),
            ],
            [DownSample(channels=1280)],  # (h/4,w/2.1280)
            [ResBlock(channels=1280, out_channels=1280)],
            [ResBlock(channels=1280, out_channels=1280)],
        ]
        self.middle_block = [
            ResBlock(channels=1280, out_channels=1280),
            SpatialTransformer(channels=1280, n_heads=8, d_head=160),
            ResBlock(channels=1280, out_channels=1280),
        ]
        self.output_blocks = [
            [ResBlock(channels=2560, out_channels=1280)],
            [ResBlock(channels=2560, out_channels=1280)],
            [ResBlock(channels=2560, out_channels=1280), UpSample(channels=1280)],
            [
                ResBlock(channels=2560, out_channels=1280),
                SpatialTransformer(channels=1280, n_heads=8, d_head=160),
            ],
            [
                ResBlock(channels=2560, out_channels=1280),
                SpatialTransformer(channels=1280, n_heads=8, d_head=160),
            ],
            [
                ResBlock(channels=1920, out_channels=1280),
                SpatialTransformer(channels=1280, n_heads=8, d_head=160),
                UpSample(channels=1280),
            ],
            [
                ResBlock(channels=1920, out_channels=640),
                SpatialTransformer(channels=640, n_heads=8, d_head=80),
            ],
            [
                ResBlock(channels=1280, out_channels=640),
                SpatialTransformer(channels=640, n_heads=8, d_head=80),
            ],
            [
                ResBlock(channels=960, out_channels=640),
                SpatialTransformer(channels=640, n_heads=8, d_head=80),
                UpSample(channels=640),
            ],
            [
                ResBlock(channels=960, out_channels=320),
                SpatialTransformer(channels=320, n_heads=8, d_head=40),
            ],
            [
                ResBlock(channels=640, out_channels=320),
                SpatialTransformer(channels=320, n_heads=8, d_head=40),
            ],
            [
                ResBlock(channels=640, out_channels=320),
                SpatialTransformer(channels=320, n_heads=8, d_head=40),
            ],
        ]
        self.out = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            keras.layers.Conv2D(filters=4, kernel_size=3, padding="same"),
        ]

    def call(self, inputs):
        x, t_emb, context = inputs
        t_emb = apply_seq(t_emb, self.time_embed)

        def apply(x, layer):
            if isinstance(layer, ResBlock):
                x = layer([x, t_emb])
            elif isinstance(layer, SpatialTransformer):
                x = layer([x, context])
            else:
                x = layer(x)
            return x

        saved_inputs = []
        for b in self.input_blocks:
            for layer in b:
                x = apply(x, layer)
            saved_inputs.append(x)

        for layer in self.middle_block:
            x = apply(x, layer)

        for b in self.output_blocks:
            x = tf.concat(values=[x, saved_inputs.pop()], axis=-1)
            for layer in b:
                x = apply(x, layer)
        return apply_seq(x, self.out)

    @staticmethod
    def get_model(
        image_height, image_width, max_text_len=77, download_weights=True
    ) -> keras.models.Model:
        context = keras.layers.Input((max_text_len, 768))
        t_emb = keras.layers.Input((320,))
        latent = keras.layers.Input((image_height, image_width, 4))
        diffusion_model = DiffusionModel()
        diffusion_model = keras.models.Model(
            [latent, t_emb, context], diffusion_model([latent, t_emb, context])
        )
        if download_weights:
            diffusion_model_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/diffusion_model.h5",
                file_hash="a5b2eea58365b18b40caee689a2e5d00f4c31dbcb4e1d58a9cf1071f55bbbd3a",
            )
            diffusion_model.load_weights(diffusion_model_weights_fpath)

        return diffusion_model
