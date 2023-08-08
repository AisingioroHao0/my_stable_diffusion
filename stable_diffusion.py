import numpy as np
from tqdm import tqdm
import math
from PIL import Image
import tensorflow as tf

from image_encoder import ImageEncoder
from diffusion_model import DiffusionModel
from text_encoder import TextTransformerModel
from clip_tokenizer import SimpleTokenizer
from constants import _UNCONDITIONAL_TOKENS, _ALPHAS_CUMPROD


MAX_TEXT_LEN = 77


class StableDiffusion:
    """
    
    """
    def __init__(
        self,
        img_height=512,
        img_width=512,
        max_text_len=77,
        jit_compile=False,
        download_weights=True,
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.img_latent_height = img_height // 8
        self.img_latent_width = img_width // 8

        self.tokenizer = SimpleTokenizer()

        self.text_encoder = TextTransformerModel.get_model(max_text_len)
        self.diffusion_model = DiffusionModel.get_model(
            self.img_latent_height, self.img_latent_width, max_text_len
        )
        self.image_encoder, self.image_decoder = ImageEncoder.get_models(
            self.img_height, self.img_width
        )
        self.image_encoder.summary(expand_nested=True)
        self.image_decoder.summary(expand_nested=True) 
        self.text_encoder.summary(expand_nested=True)
        self.diffusion_model.summary(expand_nested=True)
        
        
        if jit_compile:
            self.text_encoder.compile(jit_compile=True)
            self.diffusion_model.compile(jit_compile=True)
            self.image_decoder.compile(jit_compile=True)
            self.image_encoder.compile(jit_compile=True)

        self.dtype = tf.float32
        if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
            self.dtype = tf.float16

    def generate(
        self,
        prompt,
        negative_prompt=None,
        batch_size=1,
        num_steps=25,
        unconditional_guidance_scale=7.5,
        temperature=1,
        seed=None,
        input_image=None,
        input_mask=None,
        input_image_strength=0.5,
    ):
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer.encode(prompt)
        assert len(inputs) < 77, "Prompt is too long (should be < 77 tokens)"
        phrase = inputs + [49407] * (77 - len(inputs))
        phrase = np.array(phrase)[None].astype("int32")
        phrase = np.repeat(phrase, batch_size, axis=0)

        # Encode prompt tokens (and their positions) into a "context vector"
        pos_ids = np.array(list(range(77)))[None].astype("int32")
        pos_ids = np.repeat(pos_ids, batch_size, axis=0)
        context = self.text_encoder.predict_on_batch([phrase, pos_ids])

        input_image_tensor = None
        if input_image is not None:
            if type(input_image) is str:
                input_image = Image.open(input_image)
                input_image = input_image.resize((self.img_width, self.img_height))

            elif type(input_image) is np.ndarray:
                input_image = np.resize(
                    input_image, (self.img_height, self.img_width, input_image.shape[2])
                )

            input_image_array = np.array(input_image, dtype=np.float32)[None, ..., :3]
            input_image_tensor = tf.cast(
                (input_image_array / 255.0) * 2 - 1, self.dtype
            )

        if type(input_mask) is str:
            input_mask = Image.open(input_mask)
            input_mask = input_mask.resize((self.img_width, self.img_height))
            input_mask_array = np.array(input_mask, dtype=np.float32)[None, ..., None]
            input_mask_array = input_mask_array / 255.0

            latent_mask = input_mask.resize((self.img_width // 8, self.img_height // 8))
            latent_mask = np.array(latent_mask, dtype=np.float32)[None, ..., None]
            latent_mask = 1 - (latent_mask.astype("float") / 255.0)
            latent_mask_tensor = tf.cast(
                tf.repeat(latent_mask, batch_size, axis=0), self.dtype
            )

        # Tokenize negative prompt or use default padding tokens
        unconditional_tokens = _UNCONDITIONAL_TOKENS
        if negative_prompt is not None:
            inputs = self.tokenizer.encode(negative_prompt)
            assert (
                len(inputs) < 77
            ), "Negative prompt is too long (should be < 77 tokens)"
            unconditional_tokens = inputs + [49407] * (77 - len(inputs))

        # Encode unconditional tokens (and their positions into an
        # "unconditional context vector"
        unconditional_tokens = np.array(unconditional_tokens)[None].astype("int32")
        unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
        unconditional_context = self.text_encoder.predict_on_batch(
            [unconditional_tokens, pos_ids]
        )
        timesteps = np.arange(1, 1000, 1000 // num_steps)
        input_img_noise_t = timesteps[int(len(timesteps) * input_image_strength)]
        latent, alphas, alphas_prev = self.get_starting_parameters(
            timesteps,
            batch_size,
            seed,
            input_image=input_image_tensor,
            input_img_noise_t=input_img_noise_t,
        )

        if input_image is not None:
            timesteps = timesteps[: int(len(timesteps) * input_image_strength)]

        # Diffusion stage
        progbar = tqdm(list(enumerate(timesteps))[::-1])
        for index, timestep in progbar:
            progbar.set_description(f"{index:3d} {timestep:3d}")
            e_t = self.get_model_output(
                latent,
                timestep,
                context,
                unconditional_context,
                unconditional_guidance_scale,
                batch_size,
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            latent, pred_x0 = self.get_x_prev_and_pred_x0(
                latent, e_t, index, a_t, a_prev, temperature, seed
            )

            if input_mask is not None and input_image is not None:
                # If mask is provided, noise at current timestep will be added to input image.
                # The intermediate latent will be merged with input latent.
                latent_orgin, alphas, alphas_prev = self.get_starting_parameters(
                    timesteps,
                    batch_size,
                    seed,
                    input_image=input_image_tensor,
                    input_img_noise_t=timestep,
                )
                latent = latent_orgin * latent_mask_tensor + latent * (
                    1 - latent_mask_tensor
                )

        # Decoding stage
        decoded = self.image_decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255

        if input_mask is not None:
            # Merge inpainting output with original image
            decoded = (
                input_image_array * (1 - input_mask_array)
                + np.array(decoded) * input_mask_array
            )

        return np.clip(decoded, 0, 255).astype("uint8")

    def timestep_embedding(self, timesteps, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1), dtype=self.dtype)

    def add_noise(self, x, t, noise=None):
        """
        Add noise at timestep t to input x.
        """
        batch_size, w, h = x.shape[0], x.shape[1], x.shape[2]
        if noise is None:
            noise = tf.random.normal((batch_size, w, h, 4), dtype=self.dtype)
        sqrt_alpha_prod = _ALPHAS_CUMPROD[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - _ALPHAS_CUMPROD[t]) ** 0.5

        return sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise

    def get_starting_parameters(
        self, timesteps, batch_size, seed, input_image=None, input_img_noise_t=None
    ):
        n_h = self.img_height // 8
        n_w = self.img_width // 8
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]
        if input_image is None:
            latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
        else:
            latent = self.image_encoder(input_image)
            latent = tf.repeat(latent, batch_size, axis=0)
            latent = self.add_noise(latent, input_img_noise_t)
        return latent, alphas, alphas_prev

    def get_model_output(
        self,
        latent,
        t,
        context,
        unconditional_context,
        unconditional_guidance_scale,
        batch_size,
    ):
        timesteps = np.array([t])
        t_emb = self.timestep_embedding(timesteps)
        t_emb = np.repeat(t_emb, batch_size, axis=0)
        unconditional_latent = self.diffusion_model.predict_on_batch(
            [latent, t_emb, unconditional_context]
        )
        latent = self.diffusion_model.predict_on_batch([latent, t_emb, context])
        return unconditional_latent + unconditional_guidance_scale * (
            latent - unconditional_latent
        )

    def get_x_prev_and_pred_x0(self, x, e_t, index, a_t, a_prev, temperature, seed):
        sigma_t = 0
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
        noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0

    