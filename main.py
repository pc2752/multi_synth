import tensorflow as tf


import matplotlib.pyplot as plt

import os
import sys

# sys.path.insert(0, './griffin_lim/')
# import audio_utilities
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import h5py
import soundfile as sf
import config
from data_pipeline import data_gen
import modules_tf as modules
import utils
from reduce import mgc_to_mfsc
# import models

def one_hotize(inp, max_index=41):
    # output = np.zeros((inp.shape[0],inp.shape[1],max_index))
    # for i, index in enumerate(inp):
    #     output[i,index] = 1
    # import pdb;pdb.set_trace()
    output = np.eye(max_index)[inp.astype(int)]
    # import pdb;pdb.set_trace()
    # output = np.eye(max_index)[inp]
    return output
def binary_cross(p,q):
    return -(p * tf.log(q + 1e-12) + (1 - p) * tf.log( 1 - q + 1e-12))



def train(_):
    # stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
    # max_feat = np.array(stat_file["feats_maximus"])
    # min_feat = np.array(stat_file["feats_minimus"])
    with tf.Graph().as_default():
        

        output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,64),name='output_placeholder')

        f0_output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='f0_output_placeholder')


        f0_input_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len),name='f0_input_placeholder')
        f0_onehot_labels = tf.one_hot(indices=tf.cast(f0_input_placeholder, tf.int32), depth= len(config.notes))

        f0_context_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='f0_context_placeholder')

        phone_context_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='phone_context_placeholder')

        rand_input_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len, 64),name='rand_input_placeholder')

        prob = tf.placeholder_with_default(1.0, shape=())
        
        phoneme_labels = tf.placeholder(tf.int32, shape=(config.batch_size,config.max_phr_len),name='phoneme_placeholder')
        phone_onehot_labels = tf.one_hot(indices=tf.cast(phoneme_labels, tf.int32), depth= len(config.phonemas))



        with tf.variable_scope('Generator_feats') as scope: 
            inputs = tf.concat([phone_onehot_labels, f0_onehot_labels, phone_context_placeholder, f0_context_placeholder], axis = -1)
            voc_output = modules.GAN_generator(inputs)


        with tf.variable_scope('Discriminator_feats') as scope: 
            inputs = tf.concat([phone_onehot_labels, f0_onehot_labels, phone_context_placeholder, f0_context_placeholder], axis = -1)
            D_real = modules.GAN_discriminator((output_placeholder-0.5)*2, inputs)
            scope.reuse_variables()
            D_fake = modules.GAN_discriminator(voc_output,inputs)

        with tf.variable_scope('Generator_f0') as scope: 
            inputs = tf.concat([phone_onehot_labels, f0_onehot_labels, phone_context_placeholder, f0_context_placeholder, output_placeholder], axis = -1)
            f0_output = modules.GAN_generator_f0(inputs)
            scope.reuse_variables()
            inputs = tf.concat([phone_onehot_labels, f0_onehot_labels, phone_context_placeholder, f0_context_placeholder, (voc_output/2)+0.5], axis = -1)
            f0_output_2 = modules.GAN_generator_f0(inputs)

        with tf.variable_scope('Discriminator_f0') as scope: 
            inputs = tf.concat([phone_onehot_labels, f0_onehot_labels, phone_context_placeholder, f0_context_placeholder, output_placeholder], axis = -1)
            D_real_f0 = modules.GAN_discriminator_f0((f0_output_placeholder-0.5)*2, inputs)
            scope.reuse_variables()
            D_fake_f0 = modules.GAN_discriminator_f0(f0_output,inputs)

            scope.reuse_variables()

            inputs = tf.concat([phone_onehot_labels, f0_onehot_labels, phone_context_placeholder, f0_context_placeholder, (voc_output/2)+0.5], axis = -1)
            D_real_f0_2 = modules.GAN_discriminator_f0((f0_output_placeholder-0.5)*2, inputs)
            scope.reuse_variables()
            D_fake_f0_2 = modules.GAN_discriminator_f0(f0_output_2,inputs)

        g_params_feats = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Generator_feats")

        d_params_feats = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Discriminator_feats")

        g_params_f0 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Generator_f0")

        d_params_f0 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Discriminator_f0")


        D_loss = tf.reduce_mean(D_real +1e-12)-tf.reduce_mean(D_fake+1e-12)

        dis_summary = tf.summary.scalar('dis_loss', D_loss)

        G_loss_GAN = tf.reduce_mean(D_fake+1e-12) + tf.reduce_sum(tf.abs(output_placeholder- (voc_output/2+0.5))) *0.00005

        gen_summary = tf.summary.scalar('gen_loss', G_loss_GAN)

        D_loss_f0 = tf.reduce_mean(D_real_f0 +1e-12)-tf.reduce_mean(D_fake_f0+1e-12) + tf.reduce_mean(D_real_f0_2 +1e-12)-tf.reduce_mean(D_fake_f0_2+1e-12)

        dis_summary_f0 = tf.summary.scalar('dis_loss_f0', D_loss_f0)

        G_loss_GAN_f0 = tf.reduce_mean(D_fake_f0+1e-12) + tf.reduce_sum(tf.abs(f0_output_placeholder- (f0_output_2/2+0.5))) *0.00005 + tf.reduce_mean(D_fake_f0_2+1e-12) + tf.reduce_sum(tf.abs(f0_output_placeholder- (f0_output_2/2+0.5))) *0.00005


        gen_summary_f0 = tf.summary.scalar('gen_loss_f0', G_loss_GAN_f0)

        summary = tf.summary.merge_all()



        global_step = tf.Variable(0, name='global_step', trainable=False)

        global_step_dis = tf.Variable(0, name='global_step_dis', trainable=False)

        global_step_f0 = tf.Variable(0, name='global_step_f0', trainable=False)

        global_step_dis_f0 = tf.Variable(0, name='global_step_dis_f0', trainable=False)



        dis_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)

        gen_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)

        dis_optimizer_f0 = tf.train.RMSPropOptimizer(learning_rate=5e-5)

        gen_optimizer_f0 = tf.train.RMSPropOptimizer(learning_rate=5e-5)
        # GradientDescentOptimizer




        dis_train_function = dis_optimizer.minimize(D_loss, global_step = global_step_dis, var_list=d_params_feats)

        gen_train_function = gen_optimizer.minimize(G_loss_GAN, global_step = global_step, var_list=g_params_feats)

        dis_train_function_f0 = dis_optimizer.minimize(D_loss_f0, global_step = global_step_dis_f0, var_list=d_params_f0)

        gen_train_function_f0 = gen_optimizer.minimize(G_loss_GAN_f0, global_step = global_step, var_list=g_params_f0)

        clip_discriminator_var_op_feats = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in d_params_feats]

        clip_discriminator_var_op_f0 = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in d_params_f0]

        

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)
        sess = tf.Session()

        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(config.log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)


        train_summary_writer = tf.summary.FileWriter(config.log_dir+'train/', sess.graph)
        val_summary_writer = tf.summary.FileWriter(config.log_dir+'val/', sess.graph)

        
        start_epoch = int(sess.run(tf.train.get_global_step())/(config.batches_per_epoch_train))

        print("Start from: %d" % start_epoch)
        
        for epoch in xrange(start_epoch, config.num_epochs):

            if epoch<25 or epoch%100 == 0:
                n_critic = 25
            else:
                n_critic = 5

            data_generator = data_gen(sec_mode = 0)
            start_time = time.time()

            val_generator = data_gen(mode='val')

            batch_num = 0

            # epoch_pho_loss = 0
            epoch_gen_loss = 0
            epoch_dis_loss = 0
            epoch_gen_loss_f0 = 0
            epoch_dis_loss_f0 = 0


            with tf.variable_scope('Training'):

                for feats, conds in data_generator:
                    f0 = conds[:,:,2]
                    phones = conds[:,:,0]
                    f0_context = conds[:,:,-1:]
                    phones_context = conds[:,:,1:2]

                    

                    for critic_itr in range(n_critic):
                        feed_dict = {f0_output_placeholder: feats[:,:,-2:-1], f0_input_placeholder: f0, phoneme_labels: phones, phone_context_placeholder: phones_context,
                        f0_context_placeholder:f0_context, output_placeholder: feats[:,:,:64]}
                        sess.run([dis_train_function,dis_train_function_f0], feed_dict = feed_dict)
                        sess.run([clip_discriminator_var_op_feats, clip_discriminator_var_op_f0], feed_dict = feed_dict)

                    # feed_dict = {input_placeholder: feats, output_placeholder: feats[:,:,:-2], f0_input_placeholder: f0, rand_input_placeholder: np.random.uniform(-1.0, 1.0, size=[30,config.max_phr_len,4]),
                    # phoneme_labels:phos, singer_labels: singer_ids, phoneme_labels_shuffled:phos_shu, singer_labels_shuffled:sing_id_shu}

                    _,_, step_gen_loss, step_gen_loss_f0 = sess.run([ gen_train_function,gen_train_function_f0, G_loss_GAN, G_loss_GAN_f0], feed_dict = feed_dict)
                    # import pdb;pdb.set_trace()
                    # if step_gen_acc>0.3:
                    step_dis_loss, step_dis_loss_f0 = sess.run([D_loss, D_loss_f0], feed_dict = feed_dict)
                    # _, step_pho_loss, step_pho_acc = sess.run([pho_train_function, pho_loss, pho_acc], feed_dict= feed_dict)
                    # else: 
                        # step_dis_loss, step_dis_acc = sess.run([D_loss, D_accuracy], feed_dict = feed_dict)

                    # epoch_pho_loss+=step_pho_loss
                    # epoch_re_loss+=step_re_loss
                    epoch_gen_loss+=step_gen_loss
                    epoch_dis_loss+=step_dis_loss


                    # epoch_pho_acc+=step_pho_acc[0]
                    # epoch_gen_acc+=step_gen_acc
                    # epoch_dis_acc+=step_dis_acc
                    # epoch_dis_acc_fake+=step_dis_acc_fake



                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')
                    batch_num+=1

                # epoch_pho_loss = epoch_pho_loss/config.batches_per_epoch_train
                # epoch_re_loss = epoch_re_loss/config.batches_per_epoch_train
                epoch_gen_loss = epoch_gen_loss/config.batches_per_epoch_train
                epoch_dis_loss = epoch_dis_loss/config.batches_per_epoch_train
                # epoch_dis_acc_fake = epoch_dis_acc_fake/config.batches_per_epoch_train

                # epoch_pho_acc = epoch_pho_acc/config.batches_per_epoch_train
                # epoch_gen_acc = epoch_gen_acc/config.batches_per_epoch_train
                # epoch_dis_acc = epoch_dis_acc/config.batches_per_epoch_train
                summary_str = sess.run(summary, feed_dict=feed_dict)
            # import pdb;pdb.set_trace()
                train_summary_writer.add_summary(summary_str, epoch)
            # # summary_writer.add_summary(summary_str_val, epoch)
                train_summary_writer.flush()


            duration = time.time() - start_time

            # np.save('./ikala_eval/accuracies', f0_accs)

            if (epoch+1) % config.print_every == 0:
                print('epoch %d: Gen Loss = %.10f (%.3f sec)' % (epoch+1, epoch_gen_loss, duration))
                # print('        : Phone Accuracy = %.10f ' % (epoch_pho_acc))
                # print('        : Recon Loss = %.10f ' % (epoch_re_loss))
                # print('        : Gen Loss = %.10f ' % (epoch_gen_loss))
                # print('        : Gen Accuracy = %.10f ' % (epoch_gen_acc))
                print('        : Dis Loss = %.10f ' % (epoch_dis_loss))
                # print('        : Dis Accuracy = %.10f ' % (epoch_dis_acc))
                # print('        : Dis Accuracy Fake = %.10f ' % (epoch_dis_acc_fake))
                # print('        : Val Phone Accuracy = %.10f ' % (val_epoch_pho_acc))
                # print('        : Val Gen Loss = %.10f ' % (val_epoch_gen_loss))
                # print('        : Val Gen Accuracy = %.10f ' % (val_epoch_gen_acc))
                # print('        : Val Dis Loss = %.10f ' % (val_epoch_dis_loss))
                # print('        : Val Dis Accuracy = %.10f ' % (val_epoch_dis_acc))
                # print('        : Val Dis Accuracy Fake = %.10f ' % (val_epoch_dis_acc_fake))

            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                # utils.list_to_file(val_f0_accs,'./ikala_eval/accuracies_'+str(epoch+1)+'.txt')
                checkpoint_file = os.path.join(config.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)


def synth_file(file_name = "010.hdf5", singer_index = 0, file_path=config.wav_dir, show_plots=True, save_file = "GBO"):


    stat_file = h5py.File('./stats.hdf5', mode='r')
    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])
    with tf.Graph().as_default():
        
        output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='output_placeholder')


        f0_input_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len),name='f0_input_placeholder')
        f0_onehot_labels = tf.one_hot(indices=tf.cast(f0_input_placeholder, tf.int32), depth= len(config.notes))

        f0_context_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='f0_context_placeholder')

        phone_context_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='phone_context_placeholder')

        rand_input_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len, 64),name='rand_input_placeholder')


        # pho_input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len, 42),name='pho_input_placeholder')

        prob = tf.placeholder_with_default(1.0, shape=())
        
        phoneme_labels = tf.placeholder(tf.int32, shape=(config.batch_size,config.max_phr_len),name='phoneme_placeholder')
        phone_onehot_labels = tf.one_hot(indices=tf.cast(phoneme_labels, tf.int32), depth= len(config.phonemas))

        # singer_labels = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        # singer_onehot_labels = tf.one_hot(indices=tf.cast(singer_labels, tf.int32), depth=12)



        # with tf.variable_scope('phone_Model') as scope:
        #     # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        #     pho_logits = modules.phone_network(input_placeholder)
        #     pho_classes = tf.argmax(pho_logits, axis=-1)
        #     pho_probs = tf.nn.softmax(pho_logits)

        # with tf.variable_scope('Final_Model') as scope:
        #     voc_output = modules.final_net(singer_onehot_labels, f0_input_placeholder, phone_onehot_labels)
        #     voc_output_decoded = tf.nn.sigmoid(voc_output)
        #     scope.reuse_variables()
        #     voc_output_3 = modules.final_net(singer_onehot_labels, f0_input_placeholder, pho_probs)
        #     voc_output_3_decoded = tf.nn.sigmoid(voc_output_3)
            

        # with tf.variable_scope('singer_Model') as scope:
        #     singer_embedding, singer_logits = modules.singer_network(input_placeholder, prob)
        #     singer_classes = tf.argmax(singer_logits, axis=-1)
        #     singer_probs = tf.nn.softmax(singer_logits)


        with tf.variable_scope('Generator') as scope: 
            inputs = tf.concat([phone_onehot_labels, f0_onehot_labels, phone_context_placeholder, f0_context_placeholder, rand_input_placeholder], axis = -1)
            voc_output_2 = modules.GAN_generator(inputs)
            # scope.reuse_variables()
            # voc_output_2_2 = modules.GAN_generator(voc_output_3_decoded, singer_onehot_labels, phone_onehot_labels, f0_input_placeholder, rand_input_placeholder)


        with tf.variable_scope('Discriminator') as scope: 
            inputs = tf.concat([phone_onehot_labels, f0_onehot_labels, phone_context_placeholder, f0_context_placeholder], axis = -1)
            D_real = modules.GAN_discriminator((output_placeholder-0.5)*2, inputs)
            scope.reuse_variables()
            D_fake = modules.GAN_discriminator(voc_output_2,inputs)


        saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()

        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(config.log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        # saver.restore(sess, './log/model.ckpt-3999')

        # import pdb;pdb.set_trace()





        feat_file = h5py.File(config.feats_dir+file_name, "r")

        # speaker_file = h5py.File(config.voice_dir+speaker_file, "r")

        # feats = utils.input_to_feats('./54228_chorus.wav_ori_vocals.wav', mode = 1)



        feats = feat_file["world_feats"][()]

        feats = (feats - min_feat)/(max_feat-min_feat)

        phones = feat_file["phonemes"][()]

        notes = feat_file["notes"][()]

        phones = np.concatenate([phones, notes], axis = -1)


        # in_batches_f0, nchunks_in = utils.generate_overlapadd(f0_nor.reshape(-1,1))

        in_batches_pho, nchunks_in = utils.generate_overlapadd(phones)

        in_batches_feat, kaka = utils.generate_overlapadd(feats)

        # import pdb;pdb.set_trace()




        out_batches_feats = []



        for conds, feat in zip(in_batches_pho, in_batches_feat):
            # import pdb;pdb.set_trace()
            f0 = conds[:,:,2]
            phones = conds[:,:,0]
            f0_context = conds[:,:,-1:]
            phones_context = conds[:,:,1:2]

            feed_dict = {f0_input_placeholder: f0, phoneme_labels: phones, phone_context_placeholder: phones_context,
                        f0_context_placeholder:f0_context, rand_input_placeholder: feat[:,:,:64]}



            output_feats_gan = sess.run(voc_output_2, feed_dict = feed_dict)


            out_batches_feats.append(output_feats_gan /2 +0.5)




            # out_batches_voc_stft_phase.append(output_voc_stft_phase)



        

        out_batches_feats = np.array(out_batches_feats)
        # import pdb;pdb.set_trace()
        out_batches_feats = utils.overlapadd(out_batches_feats, nchunks_in) 



        feats = feats *(max_feat-min_feat)+min_feat

        out_batches_feats = out_batches_feats * (max_feat[-2]-min_feat[-2])+min_feat[-2]



        out_batches_feats= out_batches_feats[:len(feats)]

        plt.figure(1)
        plt.plot(out_batches_feats, label = 'Predicted F0')
        plt.plot(feats[:,-2], label = "Ground Truth F0")
        plt.legend()

        # # plt.figure(2)
        # # ax1 = plt.subplot(211)

        # # plt.imshow(feats[:,:60].T,aspect='auto',origin='lower')

        # # ax1.set_title("Ground Truth Vocoder Features", fontsize=10)

        # # ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)

        # # plt.imshow(out_batches_feats[:,:60].T,aspect='auto',origin='lower')

        # # ax2.set_title("Cross Entropy Output Vocoder Features", fontsize=10)

        plt.show()


        import pdb;pdb.set_trace()



        # out_batches_feats_gan= out_batches_feats_gan[:len(feats)]

        first_op = np.concatenate([feats[:,:-2],out_batches_feats, feats[:,-1:] ], axis = -1)

        # pho_op = np.concatenate([out_batches_feats_1,feats[:,-2:]], axis = -1)

        # gan_op = np.concatenate([out_batches_feats_gan,feats[:,-2:]], axis = -1)


        # import pdb;pdb.set_trace()
        # gan_op = np.ascontiguousarray(gan_op)

        # pho_op = np.ascontiguousarray(pho_op)

        first_op = np.ascontiguousarray(first_op)



        # if show_plots:

        #     plt.figure(1)

        #     ax1 = plt.subplot(311)

        #     plt.imshow(feats[:,:60].T,aspect='auto',origin='lower')

        #     ax1.set_title("Ground Truth Vocoder Features", fontsize=10)

        #     ax2 = plt.subplot(312, sharex = ax1, sharey = ax1)

        #     plt.imshow(out_batches_feats[:,:60].T,aspect='auto',origin='lower')

        #     ax2.set_title("Cross Entropy Output Vocoder Features", fontsize=10)

        #     ax3 =plt.subplot(313, sharex = ax1, sharey = ax1)

        #     ax3.set_title("GAN Vocoder Output Features", fontsize=10)

        #     # plt.imshow(out_batches_feats_1[:,:60].T,aspect='auto',origin='lower')
        #     #
        #     # plt.subplot(414, sharex = ax1, sharey = ax1)

        #     plt.imshow(out_batches_feats_gan[:,:60].T,aspect='auto',origin='lower')

        #     plt.figure(2)

        #     plt.subplot(211)

        #     plt.imshow(feats[:,60:-2].T,aspect='auto',origin='lower')

        #     plt.subplot(212)

        #     plt.imshow(out_batches_feats[:,-4:].T,aspect='auto',origin='lower')

        #     plt.show()

        #     save_file = input("Which files to synthesise G for GAN, B for Binary Entropy, "
        #                       "O for original, or any combination. Default is None").upper() or "N"

        # else:
        #     save_file = input("Which files to synthesise G for GAN, B for Binary Entropy, "
        #                   "O for original, or any combination. Default is all (GBO)").upper() or "GBO"

        # if "G" in save_file:

        #     utils.feats_to_audio(gan_op[:,:],file_name[:-4]+'gan_op.wav')

        #     print("GAN file saved to {}".format(os.path.join(config.val_dir,file_name[:-4]+'gan_op.wav' )))

        # if "O" in save_file:

        #     utils.feats_to_audio(feats[:, :], file_name[:-4]+'ori_op.wav')

        #     print("Originl file, resynthesized via WORLD vocoder saved to {}".format(os.path.join(config.val_dir, file_name[:-4] + 'ori_op.wav')))
            #
        # if "B" in save_file:
            # # utils.feats_to_audio(pho_op[:5000,:],file_name[:-4]+'phoop.wav')
            #
        utils.feats_to_audio(first_op ,file_name[:-4]+'_gan_op.wav')
        print("Binar cross entropy file saved to {}".format(os.path.join(config.val_dir, file_name[:-4] + 'bce_op.wav')))


        # utils.query_yes_no("Anything Else or Exit?")
        #
        # import pdb;pdb.set_trace()






if __name__ == '__main__':
    if '-l' in sys.argv:
        index_log = sys.argv.index('-l')
        import pdb;pdb.set_trace()
    if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
        print("Training")
        tf.app.run(main=train)
    elif sys.argv[1] == '-synth' or sys.argv[1] == '--synth' or sys.argv[1] == '--s' or sys.argv[1] == '-s':
        synth_file()
        # if len(sys.argv) < 3:
        #     print("Please give a file to synthesize")
        # else:
        #     file_name = sys.argv[2]
        #     if not file_name.endswith('.hdf5'):
        #         file_name = file_name + '.hdf5'
        #     if not file_name in os.listdir(config.feats_dir):
        #         print("Currently only supporting hdf5 files which are in the dataset, will be expanded later.")
        #     FLAG_PLOT = utils.query_yes_no("Plot plots?", default="yes")

        #     singer_index = config.singers.index(singer_name)
        #     synth_file(file_name, singer_index, show_plots = FLAG_PLOT)






