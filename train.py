import os
import torch
from absl import app

from config.config import *
from network.SE3Pose import SE3Pose

FLAGS = flags.FLAGS
from dataloader.dataloader import RealTrainDataset
import time
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES']='0'
def train(argv):

    s_epoch = 0

    # build dataset annd dataloader
    train_dataset = RealTrainDataset(source='', mode='train',
                                data_dir='/media/hongzhidu/data/6DoF/Datasets/mydata', per_obj='')
    # datadir: path to pose annotations

    # start training datasets sampler
    st_time = time.time()
    train_steps = FLAGS.train_steps
    global_step = train_steps * s_epoch  # record the number iteration
    train_size = train_steps * FLAGS.batch_size
    indices = []
    page_start = - train_size
    Train_stage = 'PoseNet_only'
    network = SE3Pose(Train_stage)
    network = network.to(device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                   num_workers=10, pin_memory=True)
    # network.load_state_dict(torch.load('/media/ubuntu/data/6DoF/mycode/checkpoints/model_29.pth'))
    lr = 0.0001

    optimizer = optim.AdamW(network.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer.zero_grad()   # first clear the grad

    for epoch in range(s_epoch, 150):
        # train one epoch

        if (epoch + 1) % 30 == 0:
            lr = lr * 0.5

        optimizer.param_groups[0]['lr'] = lr


        #################################
        for i, data in enumerate(train_dataloader, 1):
            output_dict, loss_dict \
                = network(obj_id=data['cat_id'].to(device),
                          PC=data['pcl_in'].to(device),
                          gt_R=data['rotation'].to(device),
                          gt_t=data['translation'].to(device),
                          gt_s=data['fsnet_scale'].to(device),
                          mean_shape=data['mean_shape'].to(device),
                          sym=data['sym_info'].to(device),
                          aug_bb=data['aug_bb'].to(device),
                          aug_rt_t=data['aug_rt_t'].to(device),
                          aug_rt_r=data['aug_rt_R'].to(device),
                          model_point=data['model_point'].to(device),
                          nocs_scale=data['nocs_scale'].to(device),
                          do_loss=True)

            fsnet_loss = loss_dict['fsnet_loss']
            fsnet_loss = sum(fsnet_loss.values())
            prop_loss = loss_dict['prop_loss']
            prop_loss = sum(prop_loss.values())
            geo_loss = loss_dict['geo_loss']
            geo_loss = sum(geo_loss.values())

            total_loss = fsnet_loss + prop_loss + geo_loss

            # backward
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #print(loss_dict)


            global_step += 1
            print(\
            "===> Epoch[{}]({}/{}): Total: {:.4f}, fs: {:.4f}, bdf: {:.4f}, prop: {:.4f}, geo: {:.4f}".format(epoch + 1, \
                                        i, train_steps, total_loss.item(), fsnet_loss.item(), 0, \
                                        prop_loss.item(), geo_loss.item()))

        if (epoch + 1) % FLAGS.save_every == 0 or (epoch + 1) == FLAGS.total_epoch:
            torch.save(network.state_dict(), '{0}/model_{1:02d}.pth'.format(FLAGS.model_save, epoch))


if __name__ == "__main__":
    app.run(train)

