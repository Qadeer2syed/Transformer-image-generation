# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import numpy as np
# import matplotlib.pyplot as plt

# # Model architecture
# class TextToImageTransformer(nn.Module):
#     def __init__(self, d_model):
#         super(TextToImageTransformer, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, 784),  # Output image size: 28x28=784
#             nn.Sigmoid()  # Assuming grayscale images
#         )

#     def forward(self, noise):
#         encoded = self.encoder(noise)
#         output = self.decoder(encoded)
#         return output

# # MNIST dataset loading and preprocessing
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_loader = DataLoader(
#     datasets.MNIST(root='./data', train=True, download=True, transform=transform),
#     batch_size=32, shuffle=True
# )

# # Define the model
# d_model = 128
# model = TextToImageTransformer(d_model)

# # Define optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()  # Binary cross-entropy loss for image generation

# def train(model, train_loader, optimizer, criterion, epochs=5):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for batch_idx, (data, _) in enumerate(train_loader):
#             optimizer.zero_grad()
#             noise = torch.randn(data.size(0), d_model)  # Generate random noise
#             data = data.view(data.size(0), -1)  # Flatten image tensor
#             output = model(noise)  # Feed noise to the model
#             loss = criterion(output, data)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             if batch_idx % 100 == 99:
#                 print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
#                 running_loss = 0.0

#                 # Display generated images
#                 generated_images = output.view(-1, 28, 28).detach().numpy()
#                 plt.figure(figsize=(10, 2))
#                 for i in range(10):  # Display the first 10 images in the batch
#                     plt.subplot(1, 10, i+1)
#                     plt.imshow(generated_images[i], cmap='gray')
#                     plt.axis('off')
#                 plt.show()

# # Start training
# train(model, train_loader, optimizer, criterion, epochs=50)
#---------------------------------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# # Model architecture
# class TextToImageTransformer(nn.Module):
#     def __init__(self, d_model):
#         super(TextToImageTransformer, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, 784),  # Output image size: 28x28=784
#             nn.Sigmoid()  # Assuming grayscale images
#         )

#     def forward(self, noise):
#         encoded = self.encoder(noise)
#         output = self.decoder(encoded)
#         return output

# # MNIST dataset loading and preprocessing
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_loader = DataLoader(
#     datasets.MNIST(root='./data', train=True, download=True, transform=transform),
#     batch_size=len(datasets.MNIST(root='./data', train=True, download=True, transform=transform)),
#     shuffle=True
# )

# # Define the model
# d_model = 128
# model = TextToImageTransformer(d_model)

# # Define optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()  # Binary cross-entropy loss for image generation

# def train(model, train_loader, optimizer, criterion, epochs=5):
#     model.train()
#     for epoch in range(1, epochs+1):
#         print(f"Epoch {epoch}/{epochs}")
#         running_loss = 0.0
#         for batch_idx, (data, _) in enumerate(train_loader):
#             optimizer.zero_grad()
#             noise = torch.randn(data.size(0), d_model)  # Generate random noise
#             data = data.view(data.size(0), -1)  # Flatten image tensor
#             output = model(noise)  # Feed noise to the model
#             loss = criterion(output, data)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             if batch_idx % 100 == 99:
#                 print(f'Epoch {epoch}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
#                 running_loss = 0.0

# # Start training
# epochs = 5
# train(model, train_loader, optimizer, criterion, epochs)

# # Display final generated image
# with torch.no_grad():
#     noise = torch.randn(1, d_model)  # Generate random noise for one image
#     output = model(noise)
#     generated_image = output.view(28, 28).numpy()

# plt.imshow(generated_image, cmap='gray')
# plt.axis('off')
# plt.show()
#--------------------------------------------------------------------------------------------------------
#START
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import numpy as np

# # Self-attention layer
# class SelfAttention(nn.Module):
#     def __init__(self, d_model, heads):
#         super(SelfAttention, self).__init__()
#         self.heads = heads
#         self.d_model = d_model
#         self.d_head = d_model // heads
        
#         # Weight matrices for query, key, value projections
#         self.query = nn.Linear(self.d_model, self.d_model)
#         self.key = nn.Linear(self.d_model, self.d_model)
#         self.value = nn.Linear(self.d_model, self.d_model)
        
#         # Output projection
#         self.output_projection = nn.Linear(self.d_model, self.d_model)
        
#     def forward(self, x):
#         batch_size = x.shape[0]
        
#         # Project inputs to query, key, and value
#         Q = self.query(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (batch_size, heads, seq_len, d_head)
#         K = self.key(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)    # (batch_size, heads, seq_len, d_head)
#         V = self.value(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (batch_size, heads, seq_len, d_head)
        
#         # Compute scaled dot-product attention scores
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)  # (batch_size, heads, seq_len, seq_len)
        
#         # Apply softmax to get attention probabilities
#         attention_probs = F.softmax(scores, dim=-1)
        
#         # Apply attention to values
#         attention_output = torch.matmul(attention_probs, V)  # (batch_size, heads, seq_len, d_head)
        
#         # Reshape attention output and perform output projection
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
#         output = self.output_projection(attention_output)  # (batch_size, seq_len, d_model)
        
#         return output

# # Model architecture with self-attention
# class TextToImageTransformerWithAttention(nn.Module):
#     def __init__(self, vocab_size, d_model, N, heads):
#         super(TextToImageTransformerWithAttention, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Embedding(vocab_size, d_model),
#             SelfAttention(d_model, heads),  # Add self-attention layer
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, 784),  # Output image size: 28x28=784
#             nn.Sigmoid()  # Assuming grayscale images
#         )
#         self.N = N
#         self.heads = heads

#     def forward(self, text):
#         encoded = self.encoder(text)
#         output = self.decoder(encoded)
#         # Reshape output to match target shape
#         output = output.view(-1, 784)
#         return output

# # MNIST dataset loading and preprocessing
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_loader = DataLoader(
#     datasets.MNIST(root='./data', train=True, download=True, transform=transform),
#     batch_size=64, shuffle=True
# )

# # Define the model
# vocab_size = 10  # Assuming 10 digits (0-9)
# d_model = 256
# N = 6
# heads = 32
# model = TextToImageTransformerWithAttention(vocab_size, d_model, N, heads)

# # Define optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()  # Binary cross-entropy loss for image generation

# # Training loop
# def train(model, train_loader, optimizer, criterion, epochs=5):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for batch_idx, (data, target) in enumerate(train_loader):
#             optimizer.zero_grad()
#             data = data.view(data.size(0), -1)  # Flatten image tensor
#             output = model(target)  # Assuming target contains text descriptions
            
#             loss = criterion(output, data)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
            
#             if batch_idx % 100 == 99:
#                 print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/100:.4f}')
#                 running_loss = 0.0

# import matplotlib.pyplot as plt

# # Define a function to generate and display images after training
# def generate_images(model, num_images=20, images_per_row=10):
#     with torch.no_grad():
#         noise = torch.randn(num_images, d_model)  # Generate random noise for the images
#         generated_images = model.decoder(noise).view(-1, 28, 28).cpu().numpy()

#         num_rows = num_images // images_per_row
#         plt.figure(figsize=(20, 2*num_rows))  # Increase figsize width and height for 10 images in each row
#         for i in range(num_images):
#             plt.subplot(num_rows, images_per_row, i+1)  # Adjust subplot arrangement
#             plt.imshow(generated_images[i], cmap='gray')
#             plt.axis('off')
#         plt.show()

# # Start training
# train(model, train_loader, optimizer, criterion, epochs=10)

# # Generate and display images after training
# generate_images(model, num_images=30, images_per_row=10)

#----------------------------end random noise----------------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt

# # Self-attention layer
# class SelfAttention(nn.Module):
#     def __init__(self, d_model, heads):
#         super(SelfAttention, self).__init__()
#         self.heads = heads
#         self.d_model = d_model
#         self.d_head = d_model // heads
        
#         # Weight matrices for query, key, value projections
#         self.query = nn.Linear(self.d_model, self.d_model)
#         self.key = nn.Linear(self.d_model, self.d_model)
#         self.value = nn.Linear(self.d_model, self.d_model)
        
#         # Output projection
#         self.output_projection = nn.Linear(self.d_model, self.d_model)
        
#     def forward(self, x):
#         batch_size = x.shape[0]
        
#         # Project inputs to query, key, and value
#         Q = self.query(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (batch_size, heads, seq_len, d_head)
#         K = self.key(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)    # (batch_size, heads, seq_len, d_head)
#         V = self.value(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (batch_size, heads, seq_len, d_head)
        
#         # Compute scaled dot-product attention scores
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # (batch_size, heads, seq_len, seq_len)
        
#         # Apply softmax to get attention probabilities
#         attention_probs = torch.softmax(scores, dim=-1)
        
#         # Apply attention to values
#         attention_output = torch.matmul(attention_probs, V)  # (batch_size, heads, seq_len, d_head)
        
#         # Reshape attention output and perform output projection
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
#         output = self.output_projection(attention_output)  # (batch_size, seq_len, d_model)
        
#         return output

# # Model architecture with self-attention
# class TextToImageTransformerWithAttention(nn.Module):
#     def __init__(self, vocab_size, d_model, N, heads):
#         super(TextToImageTransformerWithAttention, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Embedding(vocab_size, d_model),
#             SelfAttention(d_model, heads),  # Add self-attention layer
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, 784),  # Output image size: 28x28=784
#             nn.Sigmoid()  # Assuming grayscale images
#         )
#         self.N = N
#         self.heads = heads

#     def forward(self, text):
#         encoded = self.encoder(text)
#         output = self.decoder(encoded)
#         return output.view(-1, 1, 28, 28)  # Reshape output to match target image dimensions


# # Define function to convert digit text to tensor
# def text_to_tensor(label):
#     return torch.tensor(label, dtype=torch.long)


# def generate_image_from_text(model, label):
#     with torch.no_grad():
#         tensor = text_to_tensor(label)
#         tensor = tensor.unsqueeze(0)  # Unsqueeze to add batch dimension
#         output = model(tensor)
#         generated_image = output.view(28, 28).cpu().numpy()
#         return generated_image


# # Define training function
# # Training loop
# def train(model, train_loader, optimizer, criterion, epochs=1):
#     model.train()
#     for epoch in range(epochs):
#         for data, target in train_loader:
#             optimizer.zero_grad()
#             generated_images = model(target)
#             loss = criterion(generated_images, data)
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


# # Load MNIST dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
# train_loader = DataLoader(
#     datasets.MNIST(root='./data', train=True, download=True, transform=transform),
#     batch_size=64, shuffle=True
# )

# # Define the model
# vocab_size = 10  # Assuming 10 digits (0-9)
# d_model = 128
# N = 6
# heads = 32
# model = TextToImageTransformerWithAttention(vocab_size, d_model, N, heads)

# # Define optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()  # Binary cross-entropy loss for image generation

# # Train the model
# train(model, train_loader, optimizer, criterion, epochs=10)

# # Save the trained model
# torch.save(model.state_dict(), "trained_model.pth")

# # Load the trained model
# model = TextToImageTransformerWithAttention(vocab_size, d_model, N, heads)
# model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))

# # Example usage: Generate image from text description
# # Example usage: Generate image from text description
# # desired_digit = 8  # Change to the desired digit (0-9)
# # generated_image = generate_image_from_text(model, desired_digit)
# # plt.imshow(generated_image, cmap='gray')
# # plt.axis('off')
# # plt.show()
# # Generate and visualize one image for each class (0-9)
# fig, axes = plt.subplots(1, 10, figsize=(20, 2))
# for i in range(10):
#     generated_image = generate_image_from_text(model, i)
#     axes[i].imshow(generated_image, cmap='gray')
#     axes[i].axis('off')
#     axes[i].set_title(str(i))
# plt.show()

######################################################################################################################################
#Fashion MNIST noise

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Self-attention layer
class SelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_head = d_model // heads
        
        # Weight matrices for query, key, value projections
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project inputs to query, key, and value
        Q = self.query(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (batch_size, heads, seq_len, d_head)
        K = self.key(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)    # (batch_size, heads, seq_len, d_head)
        V = self.value(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (batch_size, heads, seq_len, d_head)
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)  # (batch_size, heads, seq_len, seq_len)
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_probs, V)  # (batch_size, heads, seq_len, d_head)
        
        # Reshape attention output and perform output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        output = self.output_projection(attention_output)  # (batch_size, seq_len, d_model)
        
        return output

# Model architecture with self-attention
class TextToImageTransformerWithAttention(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super(TextToImageTransformerWithAttention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            SelfAttention(d_model, heads),  # Add self-attention layer
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 784),  # Output image size: 28x28=784
            nn.Sigmoid()  # Assuming grayscale images
        )
        self.N = N
        self.heads = heads

    def forward(self, text):
        encoded = self.encoder(text)
        output = self.decoder(encoded)
        # Reshape output to match target shape
        output = output.view(-1, 784)
        return output

# MNIST dataset loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = DataLoader(
    datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

# Define the model
vocab_size = 10  # Assuming 10 digits (0-9)
d_model = 256
N = 6
heads = 128
model = TextToImageTransformerWithAttention(vocab_size, d_model, N, heads)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary cross-entropy loss for image generation

# Training loop
def train(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.view(data.size(0), -1)  # Flatten image tensor
            output = model(target)  # Assuming target contains text descriptions
            
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0

import matplotlib.pyplot as plt

# Define a function to generate and display images after training

def generate_images(model, num_images=20, images_per_row=10):
    with torch.no_grad():
        noise = torch.randn(num_images, d_model)  # Generate random noise for the images
        generated_images = model.decoder(noise).view(-1, 28, 28).cpu().numpy()

        num_rows = num_images // images_per_row
        plt.figure(figsize=(20, 2*num_rows))  # Increase figsize width and height for 10 images in each row
        for i in range(num_images):
            plt.subplot(num_rows, images_per_row, i+1)  # Adjust subplot arrangement
            plt.imshow(generated_images[i], cmap='gray')
            plt.axis('off')
        plt.show()



# Start training
train(model, train_loader, optimizer, criterion, epochs=20)

# Generate and display images after training
generate_images(model, num_images=30, images_per_row=10)

######################################################################################################################################

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt

# # Self-attention layer
# class SelfAttention(nn.Module):
#     def __init__(self, d_model, heads):
#         super(SelfAttention, self).__init__()
#         self.heads = heads
#         self.d_model = d_model
#         self.d_head = d_model // heads
        
#         # Weight matrices for query, key, value projections
#         self.query = nn.Linear(self.d_model, self.d_model)
#         self.key = nn.Linear(self.d_model, self.d_model)
#         self.value = nn.Linear(self.d_model, self.d_model)
        
#         # Output projection
#         self.output_projection = nn.Linear(self.d_model, self.d_model)
        
#     def forward(self, x):
#         batch_size = x.shape[0]
        
#         # Project inputs to query, key, and value
#         Q = self.query(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (batch_size, heads, seq_len, d_head)
#         K = self.key(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)    # (batch_size, heads, seq_len, d_head)
#         V = self.value(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # (batch_size, heads, seq_len, d_head)
        
#         # Compute scaled dot-product attention scores
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # (batch_size, heads, seq_len, seq_len)
        
#         # Apply softmax to get attention probabilities
#         attention_probs = torch.softmax(scores, dim=-1)
        
#         # Apply attention to values
#         attention_output = torch.matmul(attention_probs, V)  # (batch_size, heads, seq_len, d_head)
        
#         # Reshape attention output and perform output projection
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
#         output = self.output_projection(attention_output)  # (batch_size, seq_len, d_model)
        
#         return output

# # Model architecture with self-attention
# class TextToImageTransformerWithAttention(nn.Module):
#     def __init__(self, vocab_size, d_model, N, heads):
#         super(TextToImageTransformerWithAttention, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Embedding(vocab_size, d_model),
#             SelfAttention(d_model, heads),  # Add self-attention layer
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, 784),  # Output image size: 28x28=784
#             nn.Sigmoid()  # Assuming grayscale images
#         )
#         self.N = N
#         self.heads = heads

#     def forward(self, text):
#         encoded = self.encoder(text)
#         output = self.decoder(encoded)
#         return output.view(-1, 1, 28, 28)  # Reshape output to match target image dimensions


# # Define function to convert digit text to tensor
# def text_to_tensor(label):
#     return torch.tensor(label, dtype=torch.long)


# def generate_image_from_text(model, label):
#     with torch.no_grad():
#         tensor = text_to_tensor(label)
#         tensor = tensor.unsqueeze(0)  # Unsqueeze to add batch dimension
#         output = model(tensor)
#         generated_image = output.view(28, 28).cpu().numpy()
#         return generated_image


# # Define training function
# # Training loop
# def train(model, train_loader, optimizer, criterion, epochs=1):
#     model.train()
#     for epoch in range(epochs):
#         for data, target in train_loader:
#             optimizer.zero_grad()
#             generated_images = model(target)
#             loss = criterion(generated_images, data)
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


# # Load MNIST dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
# train_loader = DataLoader(
#     datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform),
#     batch_size=32, shuffle=True
# )

# # Define the model
# vocab_size = 10  # Assuming 10 digits (0-9)
# d_model = 256
# N = 6
# heads = 32
# model = TextToImageTransformerWithAttention(vocab_size, d_model, N, heads)

# # Define optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()  # Binary cross-entropy loss for image generation

# # Train the model
# train(model, train_loader, optimizer, criterion, epochs=10)

# # Save the trained model
# torch.save(model.state_dict(), "trained_model.pth")

# # Load the trained model
# model = TextToImageTransformerWithAttention(vocab_size, d_model, N, heads)
# model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))

# # Example usage: Generate image from text description
# # Example usage: Generate image from text description
# # desired_digit = 6  # Change to the desired digit (0-9)
# # generated_image = generate_image_from_text(model, desired_digit)
# # plt.imshow(generated_image, cmap='gray')
# # plt.axis('off')
# # plt.show()

# # Generate and visualize one image for each class (0-9)
# fig, axes = plt.subplots(1, 10, figsize=(20, 2))
# for i in range(10):
#     generated_image = generate_image_from_text(model, i)
#     axes[i].imshow(generated_image, cmap='gray')
#     axes[i].axis('off')
#     axes[i].set_title(str(i))
# plt.show()
















