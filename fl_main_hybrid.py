import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
# Assuming you have Client and Server classes properly defined
from client import Client
from server import Server

def allocate_data(dataset, n_clients):
    # This function will handle the data allocation strategy
    ### 这个函数可以用来做ood的分配
    client_datasets = random_split(dataset, [len(dataset)//n_clients]*n_clients)
    return client_datasets

# 设置超参数
n_epoch = 100
n_client = 9 #number of clients

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
e_clients1 = clients[:4]
e_clients2 = clients[4:8]
e_clients3 = clients[8:9]

server = Server(test_loader = test_loader,device=device)
e_server1 = Server(test_loader = test_loader,device=device)
e_server2 = Server(test_loader = test_loader,device=device)
e_server3 = Server(test_loader = test_loader,device=device)

for epoch in range(n_epoch):
    print(f"Epoch {epoch}:")
    client_models_and_losses1 = []
    client_models_and_losses2 = []
    client_models_and_losses3 = []

    e_server_models_and_losses = []

    # Cluster1
    for client in e_clients1:
        losses = client.train_epoch()
        models = client.get_model_params()
        client_models_and_losses1.append((models, losses))

    # Server aggregates the models and losses
    agg_losses1 = e_server1.aggregate_models_and_losses(client_models_and_losses1)
    e_models1 = e_server1.get_model_params()
    e_server_models_and_losses.append((e_models1, agg_losses1))

    # Cluster2
    for client in e_clients2:
        losses = client.train_epoch()
        models = client.get_model_params()
        client_models_and_losses2.append((models, losses))

    # Server aggregates the models and losses
    agg_losses2 = e_server2.aggregate_models_and_losses(client_models_and_losses2)
    e_models2 = e_server2.get_model_params()
    e_server_models_and_losses.append((e_models2, agg_losses2))


    # Cluster3
    for client in e_clients3:
        losses = client.train_epoch()
        models = client.get_model_params()
        client_models_and_losses3.append((models, losses))

    # Server aggregates the models and losses
    agg_losses3 = e_server3.aggregate_models_and_losses(client_models_and_losses3)
    e_models3 = e_server3.get_model_params()
    e_server_models_and_losses.append((e_models3, agg_losses3))

    
    # Aggregate all the edge server
    # Server aggregates the models and losses
    server.aggregate_models_and_losses(e_server_models_and_losses)
    server.test_model()  # Test the aggregated model
    server.save_models(epoch)


    # (Optional) You can send the aggregated model back to the clients if needed
    aggregated_gen, aggregated_dis = server.get_model_params()
    for client in clients:
        client.load_model_params(aggregated_gen, aggregated_dis)
