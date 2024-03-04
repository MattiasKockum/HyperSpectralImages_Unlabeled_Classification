import os
import subprocess
subprocess.call(["pip", "install", "wandb==0.15.11"])
subprocess.call(["wandb", "login", os.environ["WANDB_API_KEY"]])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import wandb
from sklearn.cluster import KMeans
import joblib
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import ViT3D
from dataset import SpectralDataset, custom_collate

def mask(source, mask, threshold=0.1):
    # source.shape = (X, Y, 3)
    # mask is hyperspectral
    filter_result = np.all(mask < threshold, axis=-1)
    source[filter_result] = [0, 0, 0]
    return source

def apply_gaussian_noise(batch, mean=0, std=1):
    noise = torch.randn_like(batch) * std + mean
    noisy_batch = batch + noise
    return noisy_batch


def train_log(loss, epoch, num_epochs, batch_idx, dataloader):
    wandb.log({"loss": loss.item(), "epoch": epoch})
    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")


def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


def test(model, test_loader, device, triplet_loss):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            # Contrastive learning
            negative_pairs = batch.roll(shifts=-1, dims=0).clone()
            negative_pairs[-1] = batch[0].clone()
            positive_pairs = apply_gaussian_noise(batch, std=0.1)
            # Forward pass
            embeddings_anchor = F.normalize(model(batch), dim=-1)
            embeddings_positive = F.normalize(model(positive_pairs), dim=-1)
            embeddings_negative = F.normalize(model(negative_pairs), dim=-1)
            # Contrastive loss
            loss = triplet_loss(
                    embeddings_anchor,
                    embeddings_positive,
                    embeddings_negative
            )
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    print(f"test loss : {test_loss}")
    wandb.log({"test_loss": test_loss})
    return test_loss


def train(args):
    dataset = SpectralDataset(args.train, args.pad, args.pad)
    test_dataset = SpectralDataset(args.test, args.pad, args.pad)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViT3D(
        image_size=args.pad,
        patch_width=args.patch_width,
        patch_height=args.patch_height,
        patch_depth=args.patch_depth,
        num_classes=args.num_classes,
        dim=args.dim,
        num_layers=args.num_layers,
        heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        bands=args.nb_bands
    ).to(device)

    # Contrastive loss
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Training loop

    os.environ["WANDB_NOTEBOOK_NAME"] = "ENSSAT_DATA.ipynb"
    config={
        "learning_rate": args.learning_rate,
        "architecture": "ViT3D",
        "dataset": "ENSSAT",
        "epochs": args.epochs,
        "early_stopping_at": args.early_stopping_threshold
    }
    wandb.init(project="ENSSAT_POC", config=config)

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            if batch == None:
                continue
            batch = batch.to(device)
            # Contrastive learning
            negative_pairs = batch.roll(shifts=-1, dims=0).clone()
            negative_pairs[-1] = batch[0].clone()
            positive_pairs = apply_gaussian_noise(batch, std=0.1)
            # Forward pass
            embeddings_anchor = F.normalize(model(batch), dim=-1)
            embeddings_positive = F.normalize(model(positive_pairs), dim=-1)
            embeddings_negative = F.normalize(model(negative_pairs), dim=-1)
            # Contrastive loss
            loss = triplet_loss(
                    embeddings_anchor,
                    embeddings_positive,
                    embeddings_negative
            )
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                train_log(loss, epoch, args.epochs, batch_idx, dataloader)
                test_loss = test(model, test_loader, device, triplet_loss)
                if test_loss < args.early_stopping_threshold:
                    break
        if test_loss < args.early_stopping_threshold:
            break

    print("Training of Transformer done")

    save_model(model, args.model_dir)

    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch == None:
                continue
            batch = batch.to(device)
            embedding = F.normalize(model(batch), dim=-1)
            all_embeddings.append(embedding)
            if batch_idx > 100:
                break

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings_np = all_embeddings.detach().numpy()
    all_embeddings_flattened = all_embeddings_np.reshape(all_embeddings_np.shape[0], -1)
    distortions = []
    K = range(1, 30)

    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_init="auto", random_state=0)
        cluster_labels = kmeanModel.fit_predict(all_embeddings_flattened)
        distortions.append(
            sum(np.min(
                cdist(all_embeddings_flattened, kmeanModel.cluster_centers_, 'euclidean'),
                axis=1)
            ) / len(all_embeddings_flattened))

    plt.plot(K, distortions, "b-x")
    plt.title('Distortion for elbow method')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Distortion')
    plt.grid(True)
    image_path = "distortion_plot.png"
    plt.savefig(image_path)
    wandb.log({"distortion_plot": wandb.Image(image_path)})
    plt.close()

    k = min([(i+1 , (i+1) ** 2 + (distortions[i] * len(distortions))**2) for i in range(len(distortions))], key=lambda x: x[1])[0]
    kmeanModel = KMeans(n_clusters=k, n_init="auto", random_state=0)
    cluster_labels = kmeanModel.fit_predict(all_embeddings_flattened)
    print("Saving KMeans model")
    kmeans_path = os.path.join(args.model_dir, "kmeans_model.pkl")
    joblib.dump(kmeanModel, kmeans_path)
    print("Saving clusters")
    with open(f"{args.model_dir}/cluster_labels.txt", 'w') as file:
        for label in cluster_labels:
            file.write(f"{label}\n")

    # Running the full model on the test set
    X1 = 0
    Y1 = 600
    X2 = 190
    Y2 = 740
    sub_dataset = SpectralDataset(args.test, args.pad, args.pad)
    sub_dataset.data = test_dataset.data[X1:X2, Y1:Y2, :]
    output_array = np.zeros(len(sub_dataset))
    with torch.no_grad():
        for i in tqdm(range(len(output_array))):
            embedding = model(sub_dataset[i].unsqueeze(0)).detach().numpy()
            pred = kmeanModel.predict(embedding)
            output_array[i] = pred[0]

    #test_shape = sub_dataset.data.shape
    #out_img = output_array.reshape((test_shape[0] - args.pad, test_shape[1] - args.pad)).T
    out_img = output_array.reshape((Y2 - Y1 - args.pad, X2 - X1 - args.pad)).T
    plt.imshow(out_img)
    image_path = "out_image.png"
    plt.savefig(image_path)
    wandb.log({"out_image": wandb.Image(image_path)})
    plt.clf()


    img = sub_dataset.data[:, :, [args.default_red, args.default_green, args.default_blue]]
    #img = img / img.max()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    image_path = "test.png"
    plt.savefig(image_path)
    wandb.log({"test": wandb.Image(image_path)})
    plt.clf()

    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(all_embeddings_np)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels)
    plt.title('Embeddings visualisation')
    plt.colorbar()
    image_path = "scatter.png"
    plt.savefig(image_path)
    wandb.log({"scatter": wandb.Image(image_path)})
    plt.clf()
    wandb.finish()
