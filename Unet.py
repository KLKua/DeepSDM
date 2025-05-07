import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):

    def __init__(self, num_vector, num_env, height, width):
        super().__init__()
        
        self.height = height
        self.width = width
        self.num_env = num_env
        self.num_vector = num_vector
        
        self.sigmoid = nn.Sigmoid()
    
        # for attention
        # Q part
        self.q_linear1 = nn.Linear(num_vector, 32)
        self.q_linear2 = nn.Linear(32, 16)
        self.q_linear3 = nn.Linear(16, 8)
        self.q_linear4 = nn.Linear(8, 100)
        
        # K part
        self.k_conv1 = nn.Conv2d(num_env, num_env, 3, padding = 'same', padding_mode = 'replicate', groups = num_env)
        self.k_conv2 = nn.Conv2d(num_env, num_env, 3, padding = 'same', padding_mode = 'replicate', groups = num_env)
        self.k_linear1 = nn.Linear((height * width // 16), 100)
        
        # V part
        self.v_conv1 = nn.Conv2d(num_env, num_env, 3, padding = 'same', padding_mode = 'replicate', groups = num_env)
        self.v_conv2 = nn.Conv2d(num_env, num_env, 3, padding = 'same', padding_mode = 'replicate', groups = num_env)        
        
        # Attention part (convert attention scores to different shape)
        self.a_linear1 = nn.Linear(num_env, 80)
        self.a_linear2 = nn.Linear(80, 160)
        self.a_linear3 = nn.Linear(160, 320)
        self.a_linear4 = nn.Linear(320, 640)
        
        
        self.v_fc = nn.Linear(num_vector, height*width)        
        self.vec_conv1_1 = nn.Conv2d(1, 4, 3, padding = 'same', padding_mode = 'replicate')
        self.vec_conv1_2 = nn.Conv2d(4, 8, 3, padding = 'same', padding_mode = 'replicate')
        self.vec_conv1_3 = nn.Conv2d(8, 16, 3, padding = 'same', padding_mode = 'replicate')
        self.vec_conv1_4 = nn.Conv2d(16, 16, 3, padding = 'same', padding_mode = 'replicate')
        self.vec_conv2_1 = nn.Conv2d(16, 32, 3, padding = 'same', padding_mode = 'replicate')
        self.vec_conv2_2 = nn.Conv2d(32, 32, 3, padding = 'same', padding_mode = 'replicate')
        self.vec_conv3_1 = nn.Conv2d(32, 64, 3, padding = 'same', padding_mode = 'replicate')
        self.vec_conv3_2 = nn.Conv2d(64, 64, 3, padding = 'same', padding_mode = 'replicate')
        self.vec_conv4_1 = nn.Conv2d(64, 128, 3, padding = 'same', padding_mode = 'replicate')
        self.vec_conv4_2 = nn.Conv2d(128, 128, 3, padding = 'same', padding_mode = 'replicate')

        
        self.conv1_1 = nn.Conv2d(num_env, 64, 3, padding = 'same', padding_mode = 'replicate')
        self.conv1_3 = nn.Conv2d(80, 64, 3, padding = 'same', padding_mode = 'replicate')
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding = 'same', padding_mode = 'replicate')
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding = 'same', padding_mode = 'replicate')
        self.conv2_3 = nn.Conv2d(160, 128, 3, padding = 'same', padding_mode = 'replicate')
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding = 'same', padding_mode = 'replicate')
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding = 'same', padding_mode = 'replicate')
        self.conv3_3 = nn.Conv2d(320, 256, 3, padding = 'same', padding_mode = 'replicate')
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding = 'same', padding_mode = 'replicate')
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding = 'same', padding_mode = 'replicate')
        self.conv4_3 = nn.Conv2d(640, 512, 3, padding = 'same', padding_mode = 'replicate')
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding = 'same', padding_mode = 'replicate')

        self.upconv7 = nn.ConvTranspose2d(512, 512, 2, stride = 2)
        self.conv7_1 = nn.Conv2d(512, 256, 3, padding = 'same', padding_mode = 'replicate')
        self.conv7_2 = nn.Conv2d(512, 256, 3, padding = 'same', padding_mode = 'replicate')
        self.conv7_3 = nn.Conv2d(256, 256, 3, padding = 'same', padding_mode = 'replicate')
        
        self.upconv8 = nn.ConvTranspose2d(256, 256, 2, stride = 2)
        self.conv8_1 = nn.Conv2d(256, 128, 3, padding = 'same', padding_mode = 'replicate')
        self.conv8_2 = nn.Conv2d(256, 128, 3, padding = 'same', padding_mode = 'replicate')
        self.conv8_3 = nn.Conv2d(128, 128, 3, padding = 'same', padding_mode = 'replicate')
        
        self.upconv9 = nn.ConvTranspose2d(128, 128, 2, stride = 2)
        self.conv9_1 = nn.Conv2d(128, 64, 3, padding = 'same', padding_mode = 'replicate')
        self.conv9_2 = nn.Conv2d(128, 64, 3, padding = 'same', padding_mode = 'replicate')
        self.conv9_3 = nn.Conv2d(64, 64, 3, padding = 'same', padding_mode = 'replicate')
        self.conv9_4 = nn.Conv2d(64, 64, 3, padding = 'same', padding_mode = 'replicate')
        self.conv9_5 = nn.Conv2d(64, 64, 3, padding = 'same', padding_mode = 'replicate')
        self.image1 = nn.Conv2d(64, 1, 1, padding = 'same', padding_mode = 'replicate')
    
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, image_input, biovector_input):
        
        
        # attention
        # Q
        q = biovector_input.flatten(1).to(torch.float32)
        q = self.q_linear1(q)
        q = self.q_linear2(q)
        q = self.q_linear3(q)
        q = self.q_linear4(q)
        Q = q[:, None, :]
        
        # K
        k = F.leaky_relu(self.k_conv1(image_input))
        k = F.max_pool2d(k, kernel_size = 2)
        k = F.leaky_relu(self.k_conv2(k))
        k = F.max_pool2d(k, kernel_size = 2)
        k = k.flatten(2)
        K = self.k_linear1(k)
        
        # V
        v = F.leaky_relu(self.v_conv1(image_input))
        V = F.leaky_relu(self.v_conv2(v))
        
        # t(K) * Q
        a = torch.bmm(Q, K.permute(0, 2, 1)) / (100**(1/2))
        A = F.softmax(a.squeeze(), dim = 1)

        # the linear transformation of attention score (A)
        A1 = F.softmax(self.a_linear1(A), dim = 1)
        A2 = F.softmax(self.a_linear2(A1), dim = 1)
        A3 = F.softmax(self.a_linear3(A2), dim = 1)
        A4 = F.softmax(self.a_linear4(A3), dim = 1)
        
        vec1 = biovector_input.view(-1, self.num_vector).to(torch.float32)
        vec1 = F.leaky_relu(self.v_fc(vec1))
        vec1 = vec1.view(-1, 1, self.height, self.width)
        vec1 = F.leaky_relu(self.vec_conv1_1(vec1))
        vec1 = F.leaky_relu(self.vec_conv1_2(vec1))
        vec1 = F.leaky_relu(self.vec_conv1_3(vec1))
        vec1 = F.group_norm(vec1, num_groups = 4)
        vec1 = F.leaky_relu(self.vec_conv1_4(vec1))
        vecpool1 = F.max_pool2d(vec1, kernel_size = 2)
        
        
        vec2 = F.leaky_relu(self.vec_conv2_1(vecpool1))
        vec2 = F.group_norm(vec2, num_groups = 4)
        vec2 = F.leaky_relu(self.vec_conv2_2(vec2))
        vecpool2 = F.max_pool2d(vec2, kernel_size = 2)
        
        vec3 = F.leaky_relu(self.vec_conv3_1(vecpool2))
        vec3 = F.group_norm(vec3, num_groups = 4)
        vec3 = F.leaky_relu(self.vec_conv3_2(vec3))
        vecpool3 = F.max_pool2d(vec3, kernel_size = 2)
        
        vec4 = F.leaky_relu(self.vec_conv4_1(vecpool3))
        vec4 = F.group_norm(vec4, num_groups = 4)
        vec4 = F.leaky_relu(self.vec_conv4_2(vec4))
    
        
        multiply1 = A[:, :, None, None] * V
        conv1 = F.leaky_relu(self.conv1_1(multiply1))
        conv1 = F.group_norm(conv1, num_groups = 4)
        merge1 = torch.cat([conv1, vec1], dim = 1)
        conv1 = F.leaky_relu(self.conv1_3(merge1))
        conv1 = F.group_norm(conv1, num_groups = 4)
        conv1 = F.leaky_relu(self.conv1_2(conv1))
        pool1 = F.max_pool2d(conv1, kernel_size = 2)
        
        conv2 = F.leaky_relu(self.conv2_1(pool1))
        conv2 = F.group_norm(conv2, num_groups = 4)
        merge2 = torch.cat([conv2, vec2], dim = 1)
        conv2 = F.leaky_relu(self.conv2_3(merge2))
        conv2 = F.group_norm(conv2, num_groups = 4)
        conv2 = F.leaky_relu(self.conv2_2(conv2))
        pool2 = F.max_pool2d(conv2, kernel_size = 2)
        
        conv3 = F.leaky_relu(self.conv3_1(pool2))
        conv3 = F.group_norm(conv3, num_groups = 4)
        merge3 = torch.cat([conv3, vec3], dim = 1)
        conv3 = F.leaky_relu(self.conv3_3(merge3))
        conv3 = F.group_norm(conv3, num_groups = 4)
        conv3 = F.leaky_relu(self.conv3_2(conv3))
        pool3 = F.max_pool2d(conv3, kernel_size = 2)
        
        conv4 = F.leaky_relu(self.conv4_1(pool3))
        conv4 = F.group_norm(conv4, num_groups = 4)
        merge4 = torch.cat([conv4, vec4], dim = 1)
        conv4 = F.leaky_relu(self.conv4_3(merge4))
        conv4 = F.group_norm(conv4, num_groups = 4)
        conv4 = F.leaky_relu(self.conv4_2(conv4))
        drop4 = F.dropout(conv4)
        
        
        up7 = F.interpolate(drop4, scale_factor = 2)
        conv7 = F.leaky_relu(self.conv7_1(up7))
        conv7 = F.group_norm(conv7, num_groups = 4)
        merge7 = torch.cat([conv7, conv3], dim = 1)
        conv7 = F.leaky_relu(self.conv7_2(merge7))
        conv7 = F.group_norm(conv7, num_groups = 4)
        conv7 = F.leaky_relu(self.conv7_3(conv7))
        
        up8 = F.interpolate(conv7, scale_factor = 2)
        conv8 = F.leaky_relu(self.conv8_1(up8))
        conv8 = F.group_norm(conv8, num_groups = 4)
        merge8 = torch.cat([conv8, conv2], dim = 1)
        conv8 = F.leaky_relu(self.conv8_2(merge8))
        conv8 = F.group_norm(conv8, num_groups = 4)
        conv8 = F.leaky_relu(self.conv8_3(conv8))
        
        up9 = F.interpolate(conv8, scale_factor = 2)
        conv9 = F.leaky_relu(self.conv9_1(up9))
        conv9 = F.group_norm(conv9, num_groups = 4)        
        merge9 = torch.cat([conv9, conv1], dim = 1)
        conv9 = F.leaky_relu(self.conv9_2(merge9))
        conv9 = F.group_norm(conv9, num_groups = 4)
        conv9 = F.leaky_relu(self.conv9_3(conv9))
        conv9 = F.group_norm(conv9, num_groups = 4)
        conv9 = F.leaky_relu(self.conv9_4(conv9))
        conv9 = F.leaky_relu(self.conv9_5(conv9))
#         image_output = self.sigmoid(self.image1(conv9))
        image_output = self.image1(conv9)
#         conv9 = self.image1(conv9)
#         image_output = self.batch_norm(conv9)
#         image_output = F.instance_norm(conv9)
        return image_output, A
    
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
