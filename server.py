import torch
from models import Generator, Discriminator
from collections import OrderedDict
from torch import nn
from models import Generator, Discriminator,calc_gradient_penalty
#from just_wgan import preprocess_images_for_inception, get_inception_features, calculate_frechet_distance, compute_fid
from torchvision.models import inception_v3
from torch.nn.functional import interpolate
import numpy as np
from scipy.linalg import sqrtm

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

class Server:
    def __init__(self, test_loader,device):
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.criterion = nn.BCELoss()
        self.test_loader = test_loader
        self.device = device
        self.batch_size = 50
        
        # Load the inception model
        inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        # Removing the classification head from the model
        inception_model.fc = torch.nn.Identity()
        self.inception_model = inception_model.eval()

    def aggregate_models_and_losses(self, client_models_and_losses):
        total_clients = len(client_models_and_losses)
        aggregated_g_loss = 0
        aggregated_d_loss = 0

        # Initialize aggregated models
        aggregated_model_gen = OrderedDict()
        aggregated_model_dis = OrderedDict()

        # Aggregating the models and losses
        for models, losses in client_models_and_losses:
            model_gen, model_dis = models  # Unpack the models tuple
            g_loss, d_loss = losses  # Unpack the losses tuple

            for key in model_gen.keys():
                if key not in aggregated_model_gen:
                    aggregated_model_gen[key] = torch.zeros_like(model_gen[key]).float()  # Ensure the tensor is float type
                aggregated_model_gen[key] += model_gen[key].float() / total_clients  # Ensure the tensor is float type

            for key in model_dis.keys():
                if key not in aggregated_model_dis:
                    aggregated_model_dis[key] = torch.zeros_like(model_dis[key]).float()  # Ensure the tensor is float type
                aggregated_model_dis[key] += model_dis[key].float() / total_clients  # Ensure the tensor is float type

            aggregated_g_loss += g_loss
            aggregated_d_loss += d_loss

        # Calculate average losses
        aggregated_g_loss /= total_clients
        aggregated_d_loss /= total_clients

        # Print average losses
        print(f"Train G Loss: {aggregated_g_loss}, Train D Loss: {aggregated_d_loss}")

        # Set the aggregated models to the server's models
        self.generator.load_state_dict(aggregated_model_gen)
        self.discriminator.load_state_dict(aggregated_model_dis)

        return aggregated_g_loss, aggregated_d_loss


    def save_models(self, epoch, save_dir):
        torch.save(self.generator.state_dict(), f'{save_dir}/generator_epoch_{epoch}.pth')
        torch.save(self.discriminator.state_dict(), f'{save_dir}/discriminator_epoch_{epoch}.pth')

    def get_model_params(self):
            """
            Get the model parameters from the server.
            """
            return self.generator.state_dict(), self.discriminator.state_dict()

    def test_model(self, epoch, save_filename):
        total_D_cost = 0  # 累计判别器的损失
        total_G_cost = 0  # 累计生成器的损失
        total_Wasserstein_D = 0  # 累计Wasserstein距离
        num_batches = 0  # 累计批次
        self.generator.eval()
        self.discriminator.eval()
        with torch.no_grad():
            for _, (real_data, _) in enumerate(self.test_loader):
                num_batches += 1
                real_data = real_data.to(self.device)
                # 计算真实数据的损失
                D_real = self.discriminator(real_data).mean()
                D_real = D_real.to(self.device)
                # 生成假数据
                noise = torch.randn(self.batch_size, 128).to(self.device)
                fake_data = self.generator(noise).detach()
                # 计算假数据的损失
                D_fake = self.discriminator(fake_data).mean()   
    
                # 计算梯度惩罚         
                #gradient_penalty = calc_gradient_penalty(self.discriminator, real_data.view(-1, 28*28), fake_data.view(-1, 28*28), self.device)
                gradient_penalty = 0
                
                # D_cost 是判别器的损失值，计算方式为假数据的损失 - 真实数据的损失 + 梯度惩罚
                D_cost = D_fake - D_real + gradient_penalty
                
                # Wasserstein_D 是瓦瑟斯坦距离，计算方式为真实数据的损失 - 假数据的损失
                Wasserstein_D = D_real - D_fake
            
        
                noise = torch.randn(self.batch_size, 128).to(self.device)
                fake_data = self.generator(noise).to(self.device)
                G = self.discriminator(fake_data).mean()
                
                # G_cost 是生成器的损失值，计算方式为 -G 的值
                G_cost = -G.item()

                # 更新累计的损失和时间
                total_D_cost += D_cost.item()
                total_G_cost += G_cost
                total_Wasserstein_D += Wasserstein_D.item()
        avg_D_cost = total_D_cost / num_batches
        avg_G_cost = total_G_cost / num_batches
        avg_Wasserstein_D = total_Wasserstein_D / num_batches
        print(f"Test G_cost: {avg_G_cost}, Test D_cost: {avg_D_cost}, Test Wasserstein_D: {avg_Wasserstein_D}")
        
        # Inside your training loop:
        if epoch % 10 == 0:
            with torch.no_grad():
                # Generate a batch of images
                sample = torch.randn(50, 128, device=self.device)
                generated_images = self.generator(sample)

                # Compute FID using real and generated images
                fid_value = compute_fid(self.inception_model, real_data, generated_images)
                print(f"Epoch {epoch}, FID: {fid_value}")
            with open(save_filename, 'a') as file:  # 'a' mode appends data to the file
                file.write(f"test**epoch:{epoch},D_cost:{D_cost.item()},G_cost:{G_cost},Wasserstein_D:{Wasserstein_D.item()},FID:{fid_value}\n")  # Add a newline for readability

# You may run the server as
# server = Server(z_dim)
# server.aggregate_models(received_client_models)
# server.save_models(epoch)

