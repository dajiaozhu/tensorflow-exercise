import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#from tensorflow.image import encode_jpeg
from models import Encoder, Decoder
from attention import attention, rayleigh_quotient_loss
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = tf.keras.datasets.mnist


def Auto_loss(x_real, x_predict):
    real_data = tf.reshape(x_real, [-1, 28 * 28])
    gen_data = tf.reshape(x_predict, [-1, 28 * 28])
    result = real_data - gen_data
    # gen_data = tf.transpose(gen_data, [1, 0])
    result = tf.square(result)
    loss = tf.reduce_sum(result) / (28*28)
    return loss


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = tf.reshape(X_train, [-1, 28, 28, 1])
# print("x_train type:", type(X_train))
# print("x_train shape:", X_train.shape)
db = tf.data.Dataset.from_tensor_slices((X_train, y_train))  # 将数据集转化成tensor
db = db.map(preprocess).shuffle(10000).batch(
    batchsz)  # 将数据集转化成batch_size的大小为128的数据

# print("db_type:", type(db))
db_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)



encoder = Encoder()
encoder.build(input_shape=(None, 28,28,1))
decoder = Decoder()
decoder.build(input_shape=(None, 64, 64, 3))

encoder.load_weights("./encoder_weights.ckpt")
decoder.load_weights('./decoder_weights.ckpt')

#图片的显示
# def generate_plot_image(gen_model, test_noise):
#     pre_images = gen_model(test_noise, training=False)
#     fig = plt.figure(figsize=(4, 4))
#     for i in range(pre_images.shape[0]):
#         plt.subplot(4, 4, i+1)
#         plt.imshow((pre_images[i, :, :, 0] + 1)/2, cmap='gray')
#         plt.axis('off')
#     plt.show()

for step, (x, y) in enumerate(db):
    en_res = encoder(x)
    de_res = decoder(en_res)
    de_res = tf.squeeze(de_res, axis=3)
    if step == 0:
        fig = plt.figure(figsize=(16, 8))
        for i in range(de_res.shape[0]):
            plt.subplot(16, 8, i+1)
            plt.imshow(de_res[i], cmap='gray')
            plt.axis('off')
        plt.show()



#     else:
#         # x_encoder = tf.concat([x_encoder, en_res], axis=0)
#         break
# print("x_encoder.shape:", x_encoder.shape)







# #准备encoder编码后的image
# for step, (x, y) in enumerate(db):
#     en_res = encoder(x)
#     if step == 0:
#         x_encoder = en_res
#     else:
#         x_encoder = tf.concat([x_encoder, en_res], axis=0)
# print("x_encoder.shape:", x_encoder.shape)
#     # de_res = decoder(en_res)
#     # print(de_res.shape)
#     # loss = Auto_loss(x, de_res)
#     # if step == 50:
#     #     print(step, 'loss', float(loss)
#
# x_encoder_pre = tf.reshape(x_encoder, [-1, 4, 4, 8])
# x_encoder = tf.data.Dataset.from_tensor_slices(x_encoder_pre)
# x_encoder = x_encoder.shuffle(10000).batch(128)
# print("x_encoder:", type(x_encoder))
# #聚类的结果
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
# ewa = tf.zeros([20, 128])
# atte_model = attention(8)#表示通道数是8
# #x = tf.random.normal([100, 28, 28, 1])
# atte_model.build(input_shape=(1, 4, 4, 8))
# atte_model.summary()
# for epoch in range(2):
#     for step, x in enumerate(x_encoder):
#         with tf.GradientTape() as tape:
#             # print("x[i].shape:", x[i].shape)
#             # x_ = tf.reshape(x[i], [1, x[i].shape[0], x[i].shape[1], 1])
#             ima_embedding, clu_embedding = atte_model(x)
#             # print("clu_embedding:", clu_embedding)
#             loss, ewa = rayleigh_quotient_loss(ima_embedding, clu_embedding, ewa)
#             grads = tape.gradient(loss, atte_model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, atte_model.trainable_variables))
#         if step%50 == 0:
#             print("epoch:", epoch, "step:", step, "loss:", loss)
#             print("ewa:", ewa)
#
#
# image_res, cluster_res = atte_model(x_encoder_pre)
# result = tf.reduce_sum(cluster_res, axis=0)
# print("result:", result)