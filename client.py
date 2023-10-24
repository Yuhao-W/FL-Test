import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from models import Generator, Discriminator,calc_gradient_penalty


class Client:
    def __init__(self, client_id, train_loader, epochs, device, learning_rate = 0.0001):
        self.client_id = client_id
        self.train_loader = train_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()

        self.device = device
        self.one = torch.tensor(1, dtype=torch.float).to(device)
        self.mone = self.one * -1
        self.batch_size = 50

    def train_epoch(self, epoch, savefile_name):
        total_D_cost = 0  # 累计判别器的损失
        total_G_cost = 0  # 累计生成器的损失
        total_Wasserstein_D = 0  # 累计Wasserstein距离
        total_time = 0  # 累计时间
        num_batches = 0  # 追踪处理的批次数量

        for _, (real_data, _) in enumerate(self.train_loader):
            start_time = time.time()  # 开始计时
            num_batches += 1  # 增加批次计数
            real_data = real_data.to(self.device)
            
            # (1) 更新判别器 D: 最大化 D(x) - D(G(z)) + gradient penalty
            for p in self.discriminator.parameters():
                p.requires_grad = True 

            self.discriminator.zero_grad()

            # 计算真实数据的损失
            D_real = self.discriminator(real_data).mean()
            D_real.backward(self.mone)

            # 生成假数据
            noise = torch.randn(self.batch_size, 128).to(self.device)
            fake_data = self.generator(noise).detach()

            # 计算假数据的损失
            D_fake = self.discriminator(fake_data).mean()
            D_fake.backward(self.one)

            # 计算梯度惩罚
            #print("real_data.shape: ", real_data.shape)
            #print("fake_data.shape: ", fake_data.shape)            
            gradient_penalty = calc_gradient_penalty(self.discriminator, real_data.view(-1, 28*28), fake_data.view(-1, 28*28), self.device)
            gradient_penalty.backward()

            # print("gradient_penalty: ", gradient_penalty)
            # D_cost 是判别器的损失值，计算方式为假数据的损失 - 真实数据的损失 + 梯度惩罚
            D_cost = D_fake - D_real + gradient_penalty
            
            # Wasserstein_D 是瓦瑟斯坦距离，计算方式为真实数据的损失 - 假数据的损失
            Wasserstein_D = D_real - D_fake
            
            # 更新判别器 D
            self.optimizer_d.step()

            # (2) 更新生成器 G: 最大化 D(G(z))
            for p in self.discriminator.parameters():
                p.requires_grad = False  

            self.generator.zero_grad()

            noise = torch.randn(self.batch_size, 128).to(self.device)
            fake_data = self.generator(noise)
            G = self.discriminator(fake_data).mean()
            G.backward(self.mone)
            
            # G_cost 是生成器的损失值，计算方式为 -G 的值
            G_cost = -G.item()
            self.optimizer_g.step()
            
            # 更新累计的损失和时间
            total_D_cost += D_cost.item()
            total_G_cost += G_cost
            total_Wasserstein_D += Wasserstein_D.item()

            end_time = time.time()  # 结束计时
            elapsed_time = end_time - start_time  # 计算经过的时间
            total_time += elapsed_time  # 更新累计时间

            # # 打印每批的损失和时间
            # print(f"D_cost: {D_cost.item()}, G_cost: {G_cost}, Wasserstein_D: {Wasserstein_D.item()}, Time: {elapsed_time}s")

        # 计算平均损失和时间
        avg_D_cost = total_D_cost / num_batches
        avg_G_cost = total_G_cost / num_batches
        avg_Wasserstein_D = total_Wasserstein_D / num_batches
        avg_time = total_time / num_batches

        # 打印平均损失和时间
        print(f"Client_id:{self.client_id}, Average D_cost: {avg_D_cost}, Average G_cost: {avg_G_cost}, Average Wasserstein_D: {avg_Wasserstein_D}, Average Time: {avg_time}s")
        
        with open(savefile_name, 'a') as file:  # 'a' mode appends data to the file
            file.write(f"train**epoch:{epoch},client_id:{self.client_id},D_cost:{avg_D_cost},G_cost:{avg_G_cost},Wasserstein_D:{avg_Wasserstein_D},Average Time:{avg_time}\n")  # Add a newline for readability
        
        return avg_G_cost, avg_D_cost
                
        
    def get_model_params(self):
        return self.generator.state_dict(), self.discriminator.state_dict()
    
    def load_model_params(self, gen_state_dict, dis_state_dict):
            """
            Load the model parameters from the aggregated models on the server.
            """
            self.generator.load_state_dict(gen_state_dict)
            self.discriminator.load_state_dict(dis_state_dict)

# Usage:
# client = Client(generator, discriminator, criterion, z_dim, learning_rate, train_dataset)
# client.train_epoch()
# client.get_model_params()

