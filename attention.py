import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class attention(Model):

    def __init__(self, ch):
        super(attention, self).__init__()
        #attention：
        self.f_cov = keras.layers.Conv2D(filters=ch, kernel_size=1, padding='same', activation=keras.activations.relu)
        self.g_cov = keras.layers.Conv2D(filters=ch, kernel_size=1, padding='same', activation=keras.activations.relu)
        self.h_cov = keras.layers.Conv2D(filters=ch, kernel_size=1, padding='same', activation=keras.activations.relu)
        self.r_cov = keras.layers.Conv2D(filters=ch, kernel_size=1, padding='same', activation=keras.activations.relu)
        
        
        #image_embedding:
        self.flatten = keras.layers.Flatten()
        self.ima_dense1 = keras.layers.Dense(64)
        self.ima_dense2 = keras.layers.Dense(128)
        
        #cluster_embedding:
        self.clu_dense1 = keras.layers.Dense(128, activation=keras.activations.relu)
        self.clu_dense2 = keras.layers.Dense(20) #这里暂时假设有20各类
        # self.ewa = tf.Variable(tf.zeros((64, 128)), name='ewa')  # 保存每个类的中心点，n*k,此处假设有64类


    def atte_op(self, inputs):
        #print("input.shape:", inputs.shape)
        f = self.f_cov(inputs)
        shape = f.shape
        g = self.g_cov(inputs)
        h = self.h_cov(inputs)
        #print("f.shape:", f.shape)
        #print("f.shape[0]:", f.shape[0])
        f = tf.reshape(f, [-1, f.shape[1]*f.shape[2], f.shape[-1]])  #[h*w, c]
        #print("f.shape:", f.shape)
        f = tf.transpose(f, [0, 2, 1])  #[c, h*w]
        g = tf.reshape(g, [-1, g.shape[1]*g.shape[2], g.shape[-1]])  #[h*w, c]
        h = tf.reshape(h, [-1, h.shape[1]*h.shape[2], h.shape[-1]])
        s = tf.matmul(g, f)
        #print("s.shape", s.shape)
        beta = tf.nn.softmax(s) #数据的归一化处理
        o = tf.matmul(beta, h)
        o = tf.reshape(o, shape)
        o = self.r_cov(o)
        o = 0.1*o + 0.9*inputs
        #print("o.shape:", o.shape)
        return o  #[bs, h, w, c]

    def image_embedding(self, inputs):
        inputs = self.flatten(inputs)
        #print("inputs:", inputs.shape)
        d1 = self.ima_dense1(inputs)
        d2 = self.ima_dense2(d1)
        return d2

    def cluster_embedding(self, inputs):
        atte = self.atte_op(inputs)
        atte = tf.reduce_mean(atte, axis=3)
        atte = self.flatten(atte)
        clu_d1 = self.clu_dense1(atte)
        clu_d2 = self.clu_dense2(clu_d1)
        #将clu_d2转化成one_hot形式
        b = tf.zeros([1, clu_d2.shape[1]])
        for i in range(clu_d2.shape[0]):
            if i == 0:
                res = tf.nn.softmax(clu_d2[i])
                b = b + res
            else:
                res = tf.nn.softmax(clu_d2[i])
                res = tf.reshape(res, [1, clu_d2.shape[1]])
                b = tf.concat([res, b], axis=0)
        indices = tf.argmax(b, axis=1)
        #print("indices:", indices[:50])
        one_hot = tf.one_hot(indices, depth=20)
        #print(one_hot.shape)
        return one_hot


    def call(self, inputs):
        ima_embedding = self.image_embedding(inputs)
        clu_embedding = self.cluster_embedding(inputs)
        return ima_embedding, clu_embedding


def rayleigh_quotient_loss(image_embedding, cluster_embedding, ewa):
    cluster_num = tf.argmax(cluster_embedding, axis=1)
    for i in range(image_embedding.shape[0]):  #获取batch，计算每个类的中心
        new_value = image_embedding[i, ] * \
                0.1 + ewa[cluster_num[i], ]*0.9
        # tf.set_value(ewa[cluster_num[i], ], tf.eval(new_value))  #更新中心点
        # tf.assign(ewa[cluster_num[i], ], new_value)
        # ewa[cluster_num[i]] = new_value
        #print("new_value.shape", new_value.shape)
        new_value = tf.reshape(new_value, (1, -1))
        part1 = ewa[:(cluster_num[i])]
        part2 = ewa[(cluster_num[i] + 1):] #后续要考虑边界情况
        ewa = tf.concat([part1, new_value, part2], axis=0)
        #print("ewa.shape:", ewa.shape)
    image_re = tf.reshape(image_embedding, (image_embedding.shape[0], image_embedding.shape[1], 1))  #将image_embedding扩展成m行k列1通道向量
    mean_ewa = tf.reshape(ewa, (1, tf.shape(ewa)[1], tf.shape(ewa)[0])) #将ewa装换成1行，k列，n通道
    loss = tf.log(tf.reduce_sum((image_re - mean_ewa)**2, axis=1))  #里面的平方相当于在求每个图片embedding，与每个中心点的向量的对应元素
    weight = (cluster_embedding - 0.5) * 2  #cluster_embedding表示每张图所属的类簇是一个m*n矩阵，经过这里变化，1表示不属于该类
    loss = tf.reduce_mean(tf.reduce_sum(loss * weight, axis=1))
    return loss, ewa


def main():
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    ewa = tf.zeros([20, 128])
    atte_model = attention(1)
    x = tf.random.normal([100, 28, 28, 1])
    atte_model.build(input_shape=(1, 28, 28, 1))
    atte_model.summary()
    for i in range(100):
        with tf.GradientTape() as tape:
            #print("x[i].shape:", x[i].shape)
            x_ = tf.reshape(x[i], [1, x[i].shape[0], x[i].shape[1], 1])
            ima_embedding, clu_embedding = atte_model(x_)
            #print("clu_embedding:", clu_embedding)
            loss, ewa = rayleigh_quotient_loss(ima_embedding, clu_embedding, ewa)
            grads = tape.gradient(loss, atte_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, atte_model.trainable_variables))
    print("ewa:", ewa[:10])



if __name__ == '__main__':
    main()

