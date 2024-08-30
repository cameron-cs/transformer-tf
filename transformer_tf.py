import tensorflow as tf
from keras import Model

from keras.src.layers import Dense, Dropout, LayerNormalization, Embedding
from tensorflow.python.keras import Input

# Scale Dot Product Attention
def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


# Multi-Head Attention
def multi_head_attention(d_model, num_heads):
    depth = d_model // num_heads

    def split_heads(x, batch_size):
        x = tf.reshape(x, (batch_size, -1, num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    q_input = Input(shape=(None, d_model))
    k_input = Input(shape=(None, d_model))
    v_input = Input(shape=(None, d_model))
    mask_input = Input(shape=(1, 1, None))

    batch_size = tf.shape(q_input)[0]

    q = Dense(d_model)(q_input)
    k = Dense(d_model)(k_input)
    v = Dense(d_model)(v_input)

    q = split_heads(q, batch_size)
    k = split_heads(k, batch_size)
    v = split_heads(v, batch_size)

    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask_input)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, d_model))

    output = Dense(d_model)(concat_attention)

    return Model(inputs=[q_input, k_input, v_input, mask_input], outputs=output)


# Positional Encoding
def positional_encoding(max_len, d_model):
    pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
    _2i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rads = pos / tf.pow(10000, (2 * (_2i // 2)) / tf.cast(d_model, tf.float32))

    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_positional_encoding(seq_len, d_model):
    pos_enc = positional_encoding(seq_len, d_model)
    return pos_enc[:, :seq_len, :]

# Layer Norm
def layer_norm(eps=1e-6):
    return LayerNormalization(epsilon=eps)


# Positionwise Feed Forward
def positionwise_feed_forward(d_model, dff, drop_rate=0.1):
    inputs = Input(shape=(None, d_model))
    x = Dense(dff, activation='relu')(inputs)
    x = Dropout(drop_rate)(x)
    outputs = Dense(d_model)(x)
    return Model(inputs=inputs, outputs=outputs)


# Encoder Layer
def encoder_layer(d_model, num_heads, dff, drop_rate=0.1):
    inputs = Input(shape=(None, d_model))
    mask_input = Input(shape=(1, 1, None))

    mha = multi_head_attention(d_model, num_heads)
    ffn = positionwise_feed_forward(d_model, dff, drop_rate)

    attn_output = mha([inputs, inputs, inputs, mask_input])
    attn_output = Dropout(drop_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = ffn(out1)
    ffn_output = Dropout(drop_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return Model(inputs=[inputs, mask_input], outputs=out2)


# Decoder Layer
def decoder_layer(d_model, num_heads, dff, drop_rate=0.1):
    inputs = Input(shape=(None, d_model))
    enc_output = Input(shape=(None, d_model))
    look_ahead_mask = Input(shape=(1, 1, None))
    padding_mask = Input(shape=(1, 1, None))

    mha1 = multi_head_attention(d_model, num_heads)
    mha2 = multi_head_attention(d_model, num_heads)
    ffn = positionwise_feed_forward(d_model, dff, drop_rate)

    attn1 = mha1([inputs, inputs, inputs, look_ahead_mask])
    attn1 = Dropout(drop_rate)(attn1)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn1)

    attn2 = mha2([out1, enc_output, enc_output, padding_mask])
    attn2 = Dropout(drop_rate)(attn2)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + attn2)

    ffn_output = ffn(out2)
    ffn_output = Dropout(drop_rate)(ffn_output)
    out3 = LayerNormalization(epsilon=1e-6)(out2 + ffn_output)

    return Model(inputs=[inputs, enc_output, look_ahead_mask, padding_mask], outputs=out3)


# Encoder
def encoder(vocab_size, num_layers, d_model, num_heads, dff, max_len, drop_rate=0.1):
    inputs = Input(shape=(None,))
    mask_input = Input(shape=(1, 1, None))

    x = Embedding(vocab_size, d_model)(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += get_positional_encoding(max_len, d_model)

    x = Dropout(drop_rate)(x)

    for _ in range(num_layers):
        enc_layer = encoder_layer(d_model, num_heads, dff, drop_rate)
        x = enc_layer([x, mask_input])

    return Model(inputs=[inputs, mask_input], outputs=x)


# Decoder
def decoder(vocab_size, num_layers, d_model, num_heads, dff, max_len, drop_rate=0.1):
    inputs = Input(shape=(None,))
    enc_output = Input(shape=(None, d_model))
    look_ahead_mask = Input(shape=(1, 1, None))
    padding_mask = Input(shape=(1, 1, None))

    x = Embedding(vocab_size, d_model)(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += get_positional_encoding(max_len, d_model)

    x = Dropout(drop_rate)(x)

    for _ in range(num_layers):
        dec_layer = decoder_layer(d_model, num_heads, dff, drop_rate)
        x = dec_layer([x, enc_output, look_ahead_mask, padding_mask])

    outputs = Dense(vocab_size)(x)

    return Model(inputs=[inputs, enc_output, look_ahead_mask, padding_mask], outputs=outputs)


# Transformer
def transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target,
                drop_rate=0.1):
    inputs = Input(shape=(None,))
    targets = Input(shape=(None,))

    enc_padding_mask = Input(shape=(1, 1, None))
    look_ahead_mask = Input(shape=(1, 1, None))
    dec_padding_mask = Input(shape=(1, 1, None))

    enc_output = encoder(input_vocab_size, num_layers, d_model, num_heads, dff, pe_input, drop_rate)(
        [inputs, enc_padding_mask])

    dec_output = decoder(target_vocab_size, num_layers, d_model, num_heads, dff, pe_target, drop_rate)(
        [targets, enc_output, look_ahead_mask, dec_padding_mask])

    final_output = Dense(target_vocab_size)(dec_output)

    return Model(inputs=[inputs, targets, enc_padding_mask, look_ahead_mask, dec_padding_mask], outputs=final_output)
