import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import Generator, Discriminator
import time
from torchvision.models import inception_v3
import numpy as np
from scipy.stats import entropy
from torch.nn.functional import interpolate

from scipy.linalg import sqrtm

# 定义参数
BATCH_SIZE = 50
DIM = 64
LAMBDA = 10
N_EPOCH = 20000
OUTPUT_DIM = 784

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

save_filename = "wgan_mnist_ori.txt"


# Load the inception model
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
# Removing the classification head from the model
inception_model.fc = torch.nn.Identity()
inception_model = inception_model.eval()

def preprocess_images_for_inception(imgs):
    """Upscale images to 299x299 and convert grayscale to RGB."""
    # Reshape if the images are flattened
    if len(imgs.shape) == 2:
        imgs = imgs.view(-1, 1, 28, 28)
    imgs = interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
    imgs = imgs.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
    return imgs


def get_inception_features(imgs, model):
    """Compute the features for a batch of images."""
    imgs = preprocess_images_for_inception(imgs)
    with torch.no_grad():
        features = model(imgs).detach().cpu().numpy()
    return features

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute the Frechet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_fid(model, real_images, generated_images):
    """Compute the FID between real and generated images."""
    real_features = get_inception_features(real_images, model)
    gen_features = get_inception_features(generated_images, model)
    
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    
    epsilon = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * epsilon
    sigma_gen += np.eye(sigma_gen.shape[0]) * epsilon
    
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid



# ==================生成图像的函数定义======================
def generate_image(epoch):
    noise = torch.randn(BATCH_SIZE, 128, device=device)
    with torch.no_grad():
        generated_images = netG(noise)
    img_floder = "/home/op/xinhang/synandasyn/fl_wgan_mnist/generated_images"
    save_image(generated_images.view(BATCH_SIZE, 1, 28, 28), f'{img_floder}/output_{epoch}.png')

# ==================计算梯度惩罚的函数定义======================
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size(0), real_data.nelement() // real_data.size(0)).contiguous()
    alpha = alpha.view(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# ==================加载MNIST数据集======================
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)

# 创建模型和优化器
netG = Generator().to(device)
netD = Discriminator().to(device)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

# ==================训练循环======================
one = torch.tensor(1, dtype=torch.float).to(device)
mone = one * -1


for epoch in range(N_EPOCH):
    start_time = time.time()  # 开始计时
    
    print("epoch: ", epoch)
    for _, (real_data, _) in enumerate(train_loader):
        real_data = real_data.to(device)
        
        # (1) 更新判别器 D: 最大化 D(x) - D(G(z)) + gradient penalty
        for p in netD.parameters():
            p.requires_grad = True 

        netD.zero_grad()

        # 计算真实数据的损失
        D_real = netD(real_data).mean()
        D_real.backward(mone)

        # 生成假数据
        noise = torch.randn(BATCH_SIZE, 128).to(device)
        fake_data = netG(noise).detach()

        # 计算假数据的损失
        D_fake = netD(fake_data).mean()
        D_fake.backward(one)

        # 计算梯度惩罚
        gradient_penalty = calc_gradient_penalty(netD, real_data.view(-1, 28*28), fake_data.view(-1, 28*28))
        gradient_penalty.backward()

        # D_cost 是判别器的损失值，计算方式为假数据的损失 - 真实数据的损失 + 梯度惩罚
        D_cost = D_fake - D_real + gradient_penalty
        
        # Wasserstein_D 是瓦瑟斯坦距离，计算方式为真实数据的损失 - 假数据的损失
        Wasserstein_D = D_real - D_fake
        
        # 更新判别器 D
        optimizerD.step()

        # (2) 更新生成器 G: 最大化 D(G(z))
        for p in netD.parameters():
            p.requires_grad = False  

        netG.zero_grad()

        noise = torch.randn(BATCH_SIZE, 128).to(device)
        fake_data = netG(noise)
        G = netD(fake_data).mean()
        G.backward(mone)
        
        # G_cost 是生成器的损失值，计算方式为 -G 的值
        G_cost = -G.item()
        
        optimizerG.step()

    end_time = time.time()  # 结束计时
    
    # time 是每轮迭代的时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    
    # 打印损失和时间
    print(f"epoch {epoch}: D_cost: {D_cost.item()}, G_cost: {G_cost}, Wasserstein_D: {Wasserstein_D.item()}, Time: {elapsed_time}s")
    
    # 每1000次迭代，生成和保存图片
    if epoch % 100 == 0:
        generate_image(epoch)
    
    # Inside your training loop:
    if epoch % 10 == 0:
        with torch.no_grad():
            # Generate a batch of images
            sample = torch.randn(BATCH_SIZE, 128, device=device)
            generated_images = netG(sample)

            # Compute FID using real and generated images
            fid_value = compute_fid(inception_model, real_data, generated_images)
            print(f"Epoch {epoch}, FID: {fid_value}")
        with open(save_filename, 'a') as file:  # 'a' mode appends data to the file
            file.write(f"epoch:{epoch},time:{elapsed_time},D_cost:{D_cost.item()},G_cost:{G_cost},Wasserstein_D:{Wasserstein_D.item()},FID:{fid_value}\n")  # Add a newline for readability