
# coding: utf-8

# In[1]:


import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


image_height=128
image_width=416
epoch=1
epochs=100000
batch_size=50
beta=0.2
reuse_flag=False


# In[3]:


seq_list=os.listdir('/home/SharedData/yagnesh/kitti/sequences/')
num_of_frames=[]
for seq in seq_list:
    num_of_frames.append(len(os.listdir('/home/SharedData/yagnesh/kitti/sequences/'+seq+'/'))-3)


# In[4]:


def generate_kitti_batch(batch_size):
    given_frames=np.empty(shape=[batch_size,image_height,image_width,12])
    predict_frame=np.empty(shape=[batch_size,image_height,image_width,3])
    given_poses=np.empty(shape=[batch_size,3,4,4])
    predict_pose=np.empty(shape=[batch_size,3,4])
    for element in range(batch_size):
        seq=np.random.randint(0,11)
        start=np.random.randint(0,num_of_frames[seq]-4)
        frame1=plt.imread('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+format(start,'06d')+'.png')
        frame2=plt.imread('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+format(start+1,'06d')+'.png')
        frame3=plt.imread('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+format(start+2,'06d')+'.png')
        frame4=plt.imread('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+format(start+3,'06d')+'.png')
        frame5=plt.imread('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+format(start+4,'06d')+'.png')
        given_frames[element]=np.concatenate((frame1,frame2,frame4,frame5),axis=2)
        predict_frame[element]=frame3
        poses=np.reshape(np.loadtxt('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+seq_list[seq]+'.txt'),(-1,3,4))
        pose1=poses[start]
        pose1_inv=np.transpose(pose1[:,0:3])
        pose2=poses[start+1]
        pose3=poses[start+2]
        pose4=poses[start+3]
        pose5=poses[start+4]
        pose2=np.concatenate((np.matmul(pose1_inv,pose2[:,0:3]),np.matmul(pose1_inv,np.expand_dims(pose2[:,3]-pose1[:,3],axis=1))),axis=-1)
        pose3=np.concatenate((np.matmul(pose1_inv,pose3[:,0:3]),np.matmul(pose1_inv,np.expand_dims(pose3[:,3]-pose1[:,3],axis=1))),axis=-1)
        pose4=np.concatenate((np.matmul(pose1_inv,pose4[:,0:3]),np.matmul(pose1_inv,np.expand_dims(pose4[:,3]-pose1[:,3],axis=1))),axis=-1)
        pose5=np.concatenate((np.matmul(pose1_inv,pose5[:,0:3]),np.matmul(pose1_inv,np.expand_dims(pose5[:,3]-pose1[:,3],axis=1))),axis=-1)
        pose1=np.concatenate((np.eye(3),np.zeros((3,1))),axis=-1)
        given_poses[element]=np.stack((pose1,pose2,pose4,pose5),axis=-1)
        predict_pose[element]=pose3
    return given_frames,given_poses,predict_frame,predict_pose

def generate_kitti_batch2(batch_size):
    given_frames=np.empty(shape=[batch_size,image_height,image_width,12])
    predict_frame=np.empty(shape=[batch_size,image_height,image_width,3])
    given_poses=np.empty(shape=[batch_size,3,4,4])
    predict_pose=np.empty(shape=[batch_size,3,4])
    for element in range(batch_size):
        seq=np.random.randint(0,11)
        start=np.random.randint(0,num_of_frames[seq]-4)
        frame1=plt.imread('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+format(start,'06d')+'.png')
        frame2=plt.imread('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+format(start+1,'06d')+'.png')
        frame3=plt.imread('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+format(start+2,'06d')+'.png')
        frame4=plt.imread('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+format(start+3,'06d')+'.png')
        frame5=plt.imread('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+format(start+4,'06d')+'.png')
        given_frames[element]=np.concatenate((frame1,frame2,frame3,frame5),axis=2)
        predict_frame[element]=frame4
        poses=np.reshape(np.loadtxt('/home/SharedData/yagnesh/kitti/sequences/'+seq_list[seq]+'/'+seq_list[seq]+'.txt'),(-1,3,4))
        pose1=poses[start]
        pose1_inv=np.transpose(pose1[:,0:3])
        pose2=poses[start+1]
        pose3=poses[start+2]
        pose4=poses[start+3]
        pose5=poses[start+4]
        pose2=np.concatenate((np.matmul(pose1_inv,pose2[:,0:3]),np.matmul(pose1_inv,np.expand_dims(pose2[:,3]-pose1[:,3],axis=1))),axis=-1)
        pose3=np.concatenate((np.matmul(pose1_inv,pose3[:,0:3]),np.matmul(pose1_inv,np.expand_dims(pose3[:,3]-pose1[:,3],axis=1))),axis=-1)
        pose4=np.concatenate((np.matmul(pose1_inv,pose4[:,0:3]),np.matmul(pose1_inv,np.expand_dims(pose4[:,3]-pose1[:,3],axis=1))),axis=-1)
        pose5=np.concatenate((np.matmul(pose1_inv,pose5[:,0:3]),np.matmul(pose1_inv,np.expand_dims(pose5[:,3]-pose1[:,3],axis=1))),axis=-1)
        pose1=np.concatenate((np.eye(3),np.zeros((3,1))),axis=-1)
        given_poses[element]=np.stack((pose1,pose2,pose3,pose5),axis=-1)
        predict_pose[element]=pose4
    return given_frames,given_poses,predict_frame,predict_pose

# In[6]:


with tf.device('/gpu:1'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    sess=tf.InteractiveSession(config=config)

    def U_NET_Generator(batch_size,given_frames_tensor, given_poses_tensor, predict_pose_tensor,reuse_flag):
        with tf.variable_scope(tf.get_variable_scope(),reuse=reuse_flag):
            given_poses_tensor=tf.reshape(given_poses_tensor,shape=[batch_size,48])
            predict_pose_tensor=tf.reshape(predict_pose_tensor,shape=[batch_size,12])

            w_01=tf.get_variable(name='w_01',shape=[48,4*13*768],initializer=tf.contrib.layers.xavier_initializer())
            b_01=tf.get_variable(name='b_01',shape=[4*13*768],initializer=tf.contrib.layers.xavier_initializer())
            a_01=tf.reshape(tf.matmul(given_poses_tensor,w_01)+b_01,shape=[batch_size,4,13,768])
            a_01=tf.maximum(a_01,beta*a_01)

            w_02=tf.get_variable(name='w_02',shape=[12,4*13*256],initializer=tf.contrib.layers.xavier_initializer())
            b_02=tf.get_variable(name='b_02',shape=[4*13*256],initializer=tf.contrib.layers.xavier_initializer())
            a_02=tf.reshape(tf.matmul(predict_pose_tensor,w_02)+b_02,shape=[batch_size,4,13,256])
            a_02=tf.maximum(a_02,beta*a_02)

            w_1=tf.get_variable(name='w_1',shape=[5,5,12,64],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_1=tf.get_variable(name='b_1',shape=[64],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            a_1=tf.nn.conv2d(given_frames_tensor,w_1,strides=[1,2,2,1],padding="SAME")+b_1
            a_1=tf.maximum(a_1,beta*a_1)

            w_2=tf.get_variable(name='w_2',shape=[5,5,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_2=tf.get_variable(name='b_2',shape=[128],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            a_2=tf.nn.conv2d(a_1,w_2,strides=[1,2,2,1],padding="SAME")+b_2
            a_2=tf.maximum(a_2,beta*a_2)

            w_3=tf.get_variable(name='w_3',shape=[5,5,128,256],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_3=tf.get_variable(name='b_3',shape=[256],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            a_3=tf.nn.conv2d(a_2,w_3,strides=[1,2,2,1],padding="SAME")+b_3
            a_3=tf.maximum(a_3,beta*a_3)

            w_4=tf.get_variable(name='w_4',shape=[5,5,256,512],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_4=tf.get_variable(name='b_4',shape=[512],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            a_4=tf.nn.conv2d(a_3,w_4,strides=[1,2,2,1],padding="SAME")+b_4
            a_4=tf.maximum(a_4,beta*a_4)

            w_5=tf.get_variable(name='w_5',shape=[5,5,512,1024],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_5=tf.get_variable(name='b_5',shape=[1024],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            a_5=tf.nn.conv2d(a_4,w_5,strides=[1,2,2,1],padding="SAME")+b_5
            a_5=tf.maximum(a_5,beta*a_5)

            a_5=tf.concat([a_5,a_01,a_02],axis=3)

            w_6=tf.get_variable(name='w_6',shape=[5,5,512,2048],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_6=tf.get_variable(name='b_6',shape=[512],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            a_6=tf.nn.conv2d_transpose(a_5,w_6,output_shape=[batch_size,8,26,512],strides=[1,2,2,1],padding='SAME')+b_6
            a_6=tf.maximum(a_6,beta*a_6)

            a_6=tf.concat([a_6,a_4],axis=3)

            w_7=tf.get_variable(name='w_7',shape=[5,5,256,1024],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_7=tf.get_variable(name='b_7',shape=[256],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            a_7=tf.nn.conv2d_transpose(a_6,w_7,output_shape=[batch_size,16,52,256],strides=[1,2,2,1],padding='SAME')+b_7
            a_7=tf.maximum(a_7,beta*a_7)

            a_7=tf.concat([a_7,a_3],axis=3)

            w_8=tf.get_variable(name='w_8',shape=[5,5,128,512],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_8=tf.get_variable(name='b_8',shape=[128],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            a_8=tf.nn.conv2d_transpose(a_7,w_8,output_shape=[batch_size,32,104,128],strides=[1,2,2,1],padding='SAME')+b_8
            a_8=tf.maximum(a_8,beta*a_8)

            a_8=tf.concat([a_8,a_2],axis=3)

            w_9=tf.get_variable(name='w_9',shape=[5,5,64,256],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_9=tf.get_variable(name='b_9',shape=[64],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            a_9=tf.nn.conv2d_transpose(a_8,w_9,output_shape=[batch_size,64,208,64],strides=[1,2,2,1],padding='SAME')+b_9
            a_9=tf.maximum(a_9,beta*a_9)

            a_9=tf.concat([a_9,a_1],axis=3)

            w_10=tf.get_variable(name='w_10',shape=[5,5,3,128],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_10=tf.get_variable(name='b_10',shape=[3],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            a_10=tf.nn.conv2d_transpose(a_9,w_10,output_shape=[batch_size,128,416,3],strides=[1,2,2,1],padding='SAME')+b_10
            a_10=tf.tanh(a_10)
            a_10=tf.nn.relu(a_10)

            return a_10

    given_frames_tensor=tf.placeholder(dtype=tf.float32,shape=[batch_size,image_height,image_width,12])
    given_poses_tensor=tf.placeholder(dtype=tf.float32,shape=[batch_size,3,4,4])
    predict_frame_tensor=tf.placeholder(dtype=tf.float32,shape=[batch_size,image_height,image_width,3])
    predict_pose_tensor=tf.placeholder(dtype=tf.float32,shape=[batch_size,3,4])

    predicted_frame_tensor=U_NET_Generator(batch_size,given_frames_tensor,given_poses_tensor,predict_pose_tensor,reuse_flag=reuse_flag)
    mse=tf.losses.absolute_difference(predict_frame_tensor,predicted_frame_tensor)
    adam=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(mse,var_list=tf.trainable_variables()) # Complete only after U-net is done

    sess.run(tf.global_variables_initializer())
    reuse_flag=True


saver=tf.train.Saver()


# saver.restore(sess,'/home/SharedData/yagnesh/saved/model-900')


while epoch<=epochs:
    I,P,F,D=generate_kitti_batch(batch_size)
    _,error=sess.run([adam,mse],feed_dict={given_frames_tensor:I, given_poses_tensor:P, predict_frame_tensor:F, predict_pose_tensor:D})

    if epoch%100==0:
        saver.save(sess,'/home/SharedData/yagnesh/saved/model',global_step=epoch)
        I,P,F,D=generate_kitti_batch(batch_size)
        train_error,real,gen,giv=sess.run([mse,predict_frame_tensor,predicted_frame_tensor,given_frames_tensor],feed_dict={given_frames_tensor:I, given_poses_tensor:P, predict_frame_tensor:F, predict_pose_tensor:D})
        I,P,F,D=generate_kitti_batch2(batch_size)
        cv_error,real,gen,giv=sess.run([mse,predict_frame_tensor,predicted_frame_tensor,given_frames_tensor],feed_dict={given_frames_tensor:I, given_poses_tensor:P, predict_frame_tensor:F, predict_pose_tensor:D})
        plt.imsave('/home/SharedData/yagnesh/images/'+str(epoch)+'_'+'1.png',giv[0,:,:,6:9])
        plt.imsave('/home/SharedData/yagnesh/images/'+str(epoch)+'_'+'2.png',real[0])
        plt.imsave('/home/SharedData/yagnesh/images/'+str(epoch)+'_'+'3.png',gen[0])
        plt.imsave('/home/SharedData/yagnesh/images/'+str(epoch)+'_'+'4.png',giv[0,:,:,9:12])

        print epoch,train_error,cv_error
    epoch=epoch+1
