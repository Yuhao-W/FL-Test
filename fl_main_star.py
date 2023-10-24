import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
# Assuming you have Client and Server classes properly defined
from client import Client
from server import Server
import time

start_time = time.time()

def allocate_data(dataset, n_clients):
    # This function will handle the data allocation strategy
    ### 这个函数可以用来做ood的分配
    client_datasets = random_split(dataset, [len(dataset)//n_clients]*n_clients)
    return client_datasets

# 设置超参数
n_epoch = 100
n_client = 9 #number of clients

# Directory to save trained models
save_filename = "fl_wgan_mnist_star.txt"
save_dir = "../star_models"

# 设置设备
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cpu")
# 加载和预处理数据
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_dataset, test_dataset = random_split(mnist_dataset, [int(0.9*len(mnist_dataset)), int(0.1*len(mnist_dataset))])
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

# 通过 allocate_data 函数来分配数据给客户端
client_datasets = allocate_data(train_dataset, n_client)
client_train_loaders = [DataLoader(dataset, batch_size=50, shuffle=True) for dataset in client_datasets]

# 实例化客户端和服务器
clients = [Client(client_id=i, train_loader=loader, epochs=n_epoch, device=device) for i, loader in enumerate(client_train_loaders)]
server = Server(test_loader = test_loader,device=device)

print("*****************")

for epoch in range(n_epoch):
    print(f"Epoch {epoch}:")
    client_models_and_losses = []

    for client in clients:
        losses = client.train_epoch(epoch, save_filename)
        models = client.get_model_params()
        client_models_and_losses.append((models, losses))

    # Server aggregates the models and losses
    server.aggregate_models_and_losses(client_models_and_losses)
    server.test_model(epoch, save_filename)  # Test the aggregated model
    server.save_models(epoch, save_dir)

    # (Optional) You can send the aggregated model back to the clients if needed
    aggregated_gen, aggregated_dis = server.get_model_params()
    for client in clients:
        client.load_model_params(aggregated_gen, aggregated_dis)

end_time = time.time()
running_time = end_time - start_time
print(f"Running time: {running_time} seconds")
