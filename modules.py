


"""
The modules to be used in the final model.
"""

def GAN_generator(singer_label, phones, f0_notation, rand):

    singer_label = tf.reshape(tf.layers.dense(singer_label, config.wavenet_filters, name = "g_condi"), [config.batch_size,1,1,-1], name = "g_condi_reshape")

    phones = tf.layers.dense(phones, config.wavenet_filters, name = "G_phone", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    f0_notation = tf.layers.dense(f0_notation, config.wavenet_filters, name = "G_f0", kernel_initializer=tf.random_normal_initializer(stddev=0.02))
    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    # # conds = tf.concat([phones, f0_notation], axis = -1)

    # # conds = tf.layers.dense(conds, config.wavenet_filters, name = "G_conds")

    inputs = tf.concat([phones, f0_notation, singer_label, rand], axis = -1)

    # inputs = tf.layers.dense(inputs, config.wavenet_filters, name = "G_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    # rand = tf.layers.dense(rand, config.wavenet_filters, name = "G_rand", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    conv1 =  tf.nn.relu(tf.layers.conv2d(inputs, 32, (3,1), strides=(2,1),  padding = 'same', name = "G_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    conv5 =  tf.nn.relu(tf.layers.conv2d(conv1, 64, (3,1), strides=(2,1),  padding = 'same', name = "G_5", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    # import pdb;pdb.set_trace()

    conv6 =  tf.nn.relu(tf.layers.conv2d(conv5, 128, (3,1), strides=(2,1),  padding = 'same', name = "G_6", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    conv7 = tf.nn.relu(tf.layers.conv2d(conv6, 256, (3,1), strides=(2,1),  padding = 'same', name = "G_7", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    conv8 = tf.nn.relu(tf.layers.conv2d(conv7, 512, (3,1), strides=(2,1),  padding = 'same', name = "G_8", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv1 = tf.image.resize_images(conv8, size=(8,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv1 = tf.nn.relu(tf.layers.conv2d(deconv1, 512, (3,1), strides=(1,1),  padding = 'same', name = "G_dec1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv1 = tf.concat([deconv1, conv7], axis = -1)

    deconv2 = tf.image.resize_images(deconv1, size=(16,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv2 = tf.nn.relu(tf.layers.conv2d(deconv2, 256, (3,1), strides=(1,1),  padding = 'same', name = "G_dec2", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv2 = tf.concat([deconv2, conv6], axis = -1)


    deconv3 = tf.image.resize_images(deconv2, size=(32,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv3 = tf.nn.relu(tf.layers.conv2d(deconv3, 128, (3,1), strides=(1,1),  padding = 'same', name = "G_dec3", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv3 = tf.concat([deconv3, conv5], axis = -1)


    deconv4 = tf.image.resize_images(deconv3, size=(64,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv4 = tf.nn.relu(tf.layers.conv2d(deconv4, 64, (3,1), strides=(1,1),  padding = 'same', name = "G_dec4", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv4 = tf.concat([deconv4, conv1], axis = -1)


    deconv5 = tf.image.resize_images(deconv4, size=(128,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv5 = tf.nn.relu(tf.layers.conv2d(deconv5, 64, (3,1), strides=(1,1),  padding = 'same', name = "G_dec5", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv5 = tf.concat([deconv5, inputs], axis = -1)


    # deconv1 = tf.concat([deconv2d(conv8, [config.batch_size, 8, 1, 512], name = "G_dec1"),  conv7], axis = -1)

    # deconv2 = tf.concat([deconv2d(deconv1, [config.batch_size, 16, 1, 256], name = "g_dec2") , conv6] , axis = -1)

    # deconv3 = tf.concat([deconv2d(deconv2, [config.batch_size, 32, 1, 128], name = "G_dec3"),  conv5] , axis = -1)

    # deconv4 = tf.concat([deconv2d(deconv3, [config.batch_size, 64, 1, 64], name = "G_dec4"),  conv1], axis  = -1)

    # deconv5 = tf.concat([deconv2d(deconv4, [config.batch_size, 128, 1, config.wavenet_filters], name = "G_dec5"), inputs] , axis = -1)

    # output = tf.nn.relu(tf.layers.conv2d(deconv5 , config.wavenet_filters, 1, strides=1,  padding = 'same', name = "G_o", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    output = tf.layers.conv2d(deconv5, 64, 1, strides=1,  padding = 'same', name = "G_o_2", activation = tf.nn.tanh)

    # output = tf.layers.conv2d(deconv5, 64*128, (128,1), strides=1,  padding = 'valid', name = "G_o_2", activation = tf.nn.tanh)


    output = tf.reshape(output, [config.batch_size, config.max_phr_len, -1])

    return output


def GAN_discriminator(inputs, singer_label, phones, f0_notation):

    # inputs = tf.concat([inputs, conds], axis = -1)

    singer_label = tf.reshape(tf.layers.dense(singer_label, config.wavenet_filters, name = "d_condi"), [config.batch_size,1,1,-1], name = "d_condi_reshape")

    phones = tf.layers.dense(phones, config.wavenet_filters, name = "D_phone", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    f0_notation = tf.layers.dense(f0_notation, config.wavenet_filters, name = "D_f0", kernel_initializer=tf.random_normal_initializer(stddev=0.02))
    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    # # conds = tf.concat([phones, f0_notation], axis = -1)

    # # conds = tf.layers.dense(conds, config.wavenet_filters, name = "G_conds")

    inputs = tf.concat([phones, f0_notation, singer_label, inputs], axis = -1)

    prenet_out = selu(tf.layers.dense(inputs, config.lstm_size*2, name = "d_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    prenet_out = selu(tf.layers.dense(prenet_out, config.lstm_size, name = "d_2",kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    num_block = config.wavenet_layers
    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(prenet_out, config.wavenet_filters, 1, name = "d_3")
    skips = []
    skip, residual = nr_wavenet_block(first_conv, dilation_rate=1, name = "nr_wavenet_block_0")
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1), name = "nr_wavenet_block_"+str(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,config.wavenet_filters,1, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name = "d_4")

    output = selu(output)

    output = tf.layers.conv1d(output,1,1, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name = "d_5")

    output = selu(output)

    return output