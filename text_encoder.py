import tensorflow as tf
from tensorflow import keras
import numpy as np

# 默认使用CLIP配置


class TextEmbeddingBlock(keras.layers.Layer):
    """
    文本编码块
    输入为文本的token id和位置id (batch_size,2,max_text_size),
    输出为文本的embedding编码结果 (batch_size,max_text_size,embedding_size)
    """

    def __init__(self, voc_size=49408, max_text_size=77, embedding_dim=768):
        super().__init__()
        self.token_embedding = keras.layers.Embedding(
            voc_size, embedding_dim, name="token_embedding"
        )
        self.position_embedding = keras.layers.Embedding(
            max_text_size, embedding_dim, name="position_embedding"
        )

    def call(self, inputs):
        input_ids, position_ids = inputs
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class CLIPAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.embed_dim = 768
        self.num_heads = 12
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = keras.layers.Dense(self.embed_dim)
        self.k_proj = keras.layers.Dense(self.embed_dim)
        self.v_proj = keras.layers.Dense(self.embed_dim)
        self.out_proj = keras.layers.Dense(self.embed_dim)

    def _shape(self, tensor, seq_len: int, bsz: int):
        a = tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim))
        # bs , n_head , seq_len , head_dim
        return keras.layers.Permute((2, 1, 3))(a)

    def call(self, inputs):
        hidden_states, causal_attention_mask = inputs
        bsz, tgt_len, embed_dim = hidden_states.shape
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, -1)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, -1)

        proj_shape = (-1, tgt_len, self.head_dim)
        query_states = self._shape(query_states, tgt_len, -1)
        query_states = tf.reshape(query_states, proj_shape)
        key_states = tf.reshape(key_states, proj_shape)

        src_len = tgt_len
        value_states = tf.reshape(value_states, proj_shape)
        attn_weights = query_states @ keras.layers.Permute((2, 1))(key_states)

        attn_weights = tf.reshape(attn_weights, (-1, self.num_heads, tgt_len, src_len))
        attn_weights = attn_weights + causal_attention_mask
        attn_weights = tf.reshape(attn_weights, (-1, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights)
        attn_output = attn_weights @ value_states

        attn_output = tf.reshape(
            attn_output, (-1, self.num_heads, tgt_len, self.head_dim)
        )
        attn_output = keras.layers.Permute((2, 1, 3))(attn_output)
        attn_output = tf.reshape(attn_output, (-1, tgt_len, embed_dim))

        return self.out_proj(attn_output)


class TextEncoderBlock(keras.layers.Layer):
    """
        使用attention机制生成文本编码结果
    Args:
        keras (_type_): _description_
    """

    def __init__(self, embedding_dim=768):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.multi_head_attention = CLIPAttention()

        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.dense1 = keras.layers.Dense(4 * self.embedding_dim)
        self.dense2 = keras.layers.Dense(self.embedding_dim)

    def call(self, inputs):
        hidden_states, attention_mask = inputs
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.multi_head_attention([hidden_states, attention_mask])
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        hidden_states = self.dense1(hidden_states)
        hidden_states = hidden_states * tf.sigmoid(hidden_states * 1.702)  # qicke_gelu
        hidden_states = self.dense2(hidden_states)

        return residual + hidden_states


class TextEncoder(keras.layers.Layer):
    def __init__(self, encoder_block_num=12, embedding_dim=768):
        super().__init__()
        self.layers = [
            TextEncoderBlock(embedding_dim) for i in range(encoder_block_num)
        ]

    def call(self, inputs):
        [hidden_states, causal_attention_mask] = inputs
        for layer in self.layers:
            hidden_states = layer([hidden_states, causal_attention_mask])
        return hidden_states


class TextTransformerModel(keras.models.Model):
    """
        使用EmbeddingBock得到融合位置信息的词向量
        使用TextEncoder得到文本编码
        输入为单词索引与位置索引

    Args:
        keras (_type_): _description_
    """

    def __init__(
        self, encoder_block_num=12, voc_size=49408, max_text_size=77, embedding_dim=768
    ):
        super().__init__()
        self.embeddings = TextEmbeddingBlock(
            voc_size=voc_size, max_text_size=max_text_size, embedding_dim=embedding_dim
        )
        self.encoder = TextEncoder(encoder_block_num, embedding_dim=embedding_dim)
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.causal_attention_mask = tf.constant(
            np.triu(np.ones((1, 1, 77, 77), dtype="float32") * -np.inf, k=1)
        )
        # self.causal_attention_mask = tf.linalg.band_part(tf.ones((1, 1, 77, 77))*-tf.inf, -1, 0)

    def call(self, inputs):
        input_ids, position_ids = inputs
        x = self.embeddings([input_ids, position_ids])
        x = self.encoder([x, self.causal_attention_mask])
        return self.final_layer_norm(x)

    @staticmethod
    def get_model(max_text_len=77, download_weights=True) -> keras.models.Model:
        input_word_ids = keras.layers.Input(shape=(max_text_len,), dtype="int32")
        input_pos_ids = keras.layers.Input(shape=(max_text_len,), dtype="int32")
        embeds = TextTransformerModel()([input_word_ids, input_pos_ids])
        text_encoder = keras.models.Model([input_word_ids, input_pos_ids], embeds)

        text_encoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/text_encoder.h5",
            file_hash="d7805118aeb156fc1d39e38a9a082b05501e2af8c8fbdc1753c9cb85212d6619",
        )
        text_encoder.load_weights(text_encoder_weights_fpath)
        return text_encoder
