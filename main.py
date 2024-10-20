import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from tensorboardX import SummaryWriter
import wandb
from tqdm import tqdm
import yaml
import random
import logging
from collections import deque, defaultdict
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reverse_complement(seq: str) -> str:
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

class DNADataset(Dataset):
    def __init__(self, data_path: str, seq_length: int, augment: bool = False):
        self.data = np.load(data_path)
        self.seq_length = seq_length
        self.nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.data[idx]
        if self.augment and random.random() < 0.5:
            seq = reverse_complement(seq)
        seq = [self.nucleotide_map.get(nt, 0) for nt in seq]
        return torch.tensor(seq, dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MaskedDiffusionModel(nn.Module):
    def __init__(self, seq_length: int, vocab_size: int, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size + 1, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size*4, dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mask = torch.bernoulli(torch.full_like(x, t.item(), dtype=torch.float)).bool()
        x_masked = x.clone()
        x_masked[mask] = self.vocab_size
        embedded = self.embedding(x_masked)
        encoded = self.positional_encoding(embedded)
        time_embed = self.time_embedding(t).unsqueeze(1).expand(-1, self.seq_length, -1)
        encoded += time_embed
        encoded = self.layer_norm(encoded)
        output = self.transformer_encoder(encoded.transpose(0, 1)).transpose(0, 1)
        logits = self.fc(output)
        return logits

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def initial_distribution(self) -> torch.Tensor:
        return torch.ones(self.vocab_size) / self.vocab_size

    def transition_rate(self, t: torch.Tensor) -> torch.Tensor:
        gamma = 1 / (1 - t)
        Q = -gamma * torch.eye(self.vocab_size + 1, device=t.device)
        Q[-1, :-1] = gamma / self.vocab_size
        Q[:-1, -1] = gamma
        return Q[:-1, :-1]

class RewardModel(nn.Module):
    def __init__(self, seq_length: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size*4, dropout=0.1, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        encoded = self.positional_encoding(embedded)
        output = self.transformer_encoder(encoded.transpose(0, 1)).transpose(0, 1)
        return self.fc(output.mean(dim=1))

class Discriminator(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_len=5000)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size*4, dropout=0.1, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        encoded = self.positional_encoding(embedded)
        output = self.transformer_encoder(encoded.transpose(0, 1)).transpose(0, 1)
        return self.fc(output.mean(dim=1))

class AttentionVisualizer:
    def __init__(self, model: 'DRAKES', writer: SummaryWriter, device: torch.device):
        self.model = model
        self.writer = writer
        self.device = device

    def visualize_attention(self, x: torch.Tensor, layer: int = 0):
        self.model.pretrained_model.eval()
        with torch.no_grad():
            embedded = self.model.pretrained_model.embedding(x)
            encoded = self.model.pretrained_model.positional_encoding(embedded)
            encoded = self.model.pretrained_model.layer_norm(encoded)
            for i, layer_module in enumerate(self.model.pretrained_model.transformer_encoder.layers):
                if i == layer:
                    attn_weights = layer_module.self_attn(encoded.transpose(0, 1), encoded.transpose(0, 1), encoded.transpose(0, 1))[1]
                    avg_attn = attn_weights.mean(dim=1).cpu().numpy()
                    avg_attn = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-8)
                    avg_attn_img = torch.tensor(avg_attn).unsqueeze(1)
                    self.writer.add_images(f'Attention/Layer_{layer}', avg_attn_img, 0)
                encoded = layer_module(encoded.transpose(0, 1)).transpose(0, 1)
            self.writer.flush()

class DRAKES:
    def __init__(self, pretrained_model: nn.Module, reward_model: nn.Module, alpha: float, learning_rate: float, batch_size: int, num_iterations: int, time_steps: int, delta_t: float, temperature: float, device: torch.device, scheduler_type: str = 'onecycle', reward_window: int = 100, diversity_weight: float = 0.01):
        self.pretrained_model = pretrained_model
        self.reward_model = reward_model
        self.alpha_initial = alpha
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.time_steps = time_steps
        self.delta_t = delta_t
        self.temperature = temperature
        self.device = device
        self.optimizer = torch.optim.AdamW(self.pretrained_model.parameters(), lr=learning_rate, weight_decay=0.01)
        if scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(self.optimizer, max_lr=learning_rate, total_steps=num_iterations)
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iterations)
        elif scheduler_type == 'reduceonplateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        else:
            raise ValueError("Unsupported scheduler type")
        self.scaler = GradScaler()
        self.writer = SummaryWriter()
        wandb.init(project="drakes_dna_design", config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "time_steps": time_steps,
            "delta_t": delta_t,
            "temperature": temperature,
            "alpha": alpha,
            "scheduler_type": scheduler_type,
            "reward_window": reward_window,
            "diversity_weight": diversity_weight
        })
        self.reward_window = reward_window
        self.recent_rewards = deque(maxlen=reward_window)
        self.diversity_weight = diversity_weight
        self.memory_bank = deque(maxlen=1000)

    def gumbel_softmax(self, logits: torch.Tensor, tau: float, hard: bool = False, dim: int = -1) -> torch.Tensor:
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim)
        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def sample_trajectory(self) -> torch.Tensor:
        x_trajectory = []
        x_t = self.pretrained_model.initial_distribution().repeat(self.batch_size, self.pretrained_model.seq_length, 1).to(self.device)
        x_trajectory.append(x_t)
        for t in range(1, self.time_steps + 1):
            current_time = torch.full((self.batch_size,), t * self.delta_t, device=self.device)
            logits = self.pretrained_model(x_t.argmax(dim=-1), current_time)
            pi_t = x_t + self.delta_t * torch.matmul(x_t, self.pretrained_model.transition_rate(current_time))
            x_t = self.gumbel_softmax(pi_t, tau=self.temperature / (t * self.delta_t), hard=True)
            x_trajectory.append(x_t)
        return torch.stack(x_trajectory)

    def compute_kl_divergence(self, x_trajectory: torch.Tensor) -> torch.Tensor:
        kl_div = 0
        for t in range(1, self.time_steps + 1):
            current_time = torch.full((self.batch_size,), t * self.delta_t, device=self.device)
            Q_theta = self.pretrained_model.transition_rate(current_time)
            Q_theta_pre = self.pretrained_model.transition_rate(current_time)
            x_t = x_trajectory[t-1]
            kl_t = torch.sum(x_t * torch.sum(Q_theta * (torch.log(Q_theta + 1e-10) - torch.log(Q_theta_pre + 1e-10)), dim=-1), dim=(1, 2))
            kl_div += kl_t * self.delta_t
        return kl_div.mean()

    def compute_diversity_loss(self, generated_sequences: torch.Tensor) -> torch.Tensor:
        batch_size = generated_sequences.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=self.device)
        dist_matrix = torch.cdist(generated_sequences.float(), generated_sequences.float(), p=1)
        dist_matrix = dist_matrix / self.pretrained_model.seq_length
        mask = torch.eye(batch_size, device=self.device).bool()
        dist_matrix = dist_matrix.masked_fill(mask, 0.0)
        avg_dist = dist_matrix.sum() / (batch_size * (batch_size - 1))
        diversity_loss = -avg_dist
        return diversity_loss

    def adjust_alpha(self) -> float:
        if len(self.recent_rewards) == 0:
            return self.alpha_initial
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
        new_alpha = max(self.alpha_initial / (avg_reward + 1e-8), 1e-4)
        return new_alpha

    def add_to_memory_bank(self, sequences: torch.Tensor, rewards: torch.Tensor):
        for seq, reward in zip(sequences, rewards):
            if reward > torch.mean(torch.tensor(list(self.recent_rewards))):
                self.memory_bank.append(seq.cpu())

    def compute_adversarial_loss(self, generated_sequences: torch.Tensor, discriminator: nn.Module) -> torch.Tensor:
        logits = discriminator(generated_sequences)
        adversarial_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        return adversarial_loss

    def compute_motif_attention(self, generated_sequences: torch.Tensor, motifs: List[str]) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)

    def compute_constraint_loss(self, generated_sequences: torch.Tensor, constraints: dict) -> torch.Tensor:
        loss = 0.0
        if 'gc_content' in constraints:
            gc_target = constraints['gc_content']
            gc_counts = ((generated_sequences == 1) | (generated_sequences == 2)).float().sum(dim=1)
            gc_content = gc_counts / self.pretrained_model.seq_length
            loss += F.mse_loss(gc_content, torch.full_like(gc_content, gc_target))
        if 'motifs' in constraints:
            motifs = constraints['motifs']
            motif_loss = self.compute_motif_attention(generated_sequences, motifs)
            loss += motif_loss
        return loss

    def train(self, train_loader: DataLoader, val_loader: DataLoader, discriminator: Optional[nn.Module] = None, motifs: Optional[List[str]] = None, constraints: Optional[dict] = None):
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        for iteration in range(1, self.num_iterations + 1):
            self.pretrained_model.train()
            train_loss = 0
            train_reward = 0
            train_kl_div = 0
            train_diversity = 0
            adversarial_loss_total = 0
            constraint_loss_total = 0
            for batch in tqdm(train_loader, desc=f"Iteration {iteration}/{self.num_iterations}"):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                with autocast():
                    x_trajectory = self.sample_trajectory()
                    generated_sequences = x_trajectory[-1].argmax(dim=-1)
                    reward = self.reward_model(generated_sequences)
                    self.recent_rewards.append(reward.mean().item())
                    current_alpha = self.adjust_alpha()
                    kl_div = self.compute_kl_divergence(x_trajectory)
                    diversity_loss = self.compute_diversity_loss(generated_sequences)
                    constraint_loss = 0.0
                    if constraints:
                        constraint_loss = self.compute_constraint_loss(generated_sequences, constraints)
                    adversarial_loss = 0.0
                    if discriminator is not None:
                        adversarial_loss = self.compute_adversarial_loss(generated_sequences, discriminator)
                    loss = -reward.mean() + current_alpha * kl_div + self.diversity_weight * diversity_loss + constraint_loss
                    if discriminator is not None:
                        loss += adversarial_loss
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.pretrained_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                train_loss += loss.item()
                train_reward += reward.mean().item()
                train_kl_div += kl_div.item()
                train_diversity += diversity_loss.item()
                adversarial_loss_total += adversarial_loss.item()
                constraint_loss_total += constraint_loss.item()
                self.add_to_memory_bank(generated_sequences, reward)
            train_loss /= len(train_loader)
            train_reward /= len(train_loader)
            train_kl_div /= len(train_loader)
            train_diversity /= len(train_loader)
            adversarial_loss_avg = adversarial_loss_total / len(train_loader) if discriminator is not None else 0.0
            constraint_loss_avg = constraint_loss_total / len(train_loader) if constraints else 0.0
            val_loss, val_reward, val_kl_div, val_diversity = self.validate(val_loader, constraints=constraints)
            self.writer.add_scalar('Loss/train', train_loss, iteration)
            self.writer.add_scalar('Reward/train', train_reward, iteration)
            self.writer.add_scalar('KL_Divergence/train', train_kl_div, iteration)
            self.writer.add_scalar('Diversity/train', train_diversity, iteration)
            self.writer.add_scalar('Loss/val', val_loss, iteration)
            self.writer.add_scalar('Reward/val', val_reward, iteration)
            self.writer.add_scalar('KL_Divergence/val', val_kl_div, iteration)
            self.writer.add_scalar('Diversity/val', val_diversity, iteration)
            self.writer.add_scalar('Constraint_Loss/train', constraint_loss_avg, iteration)
            if discriminator is not None:
                self.writer.add_scalar('Adversarial_Loss/train', adversarial_loss_avg, iteration)
            log_dict = {
                "Iteration": iteration,
                "train_loss": train_loss,
                "train_reward": train_reward,
                "train_kl_div": train_kl_div,
                "train_diversity": train_diversity,
                "train_constraint_loss": constraint_loss_avg,
                "val_loss": val_loss,
                "val_reward": val_reward,
                "val_kl_div": val_kl_div,
                "val_diversity": val_diversity
            }
            if discriminator is not None:
                log_dict["adversarial_loss"] = adversarial_loss_avg
            wandb.log(log_dict)
            logger.info(f"Iteration {iteration}: Train Loss={train_loss:.4f}, Train Reward={train_reward:.4f}, Train KL Div={train_kl_div:.4f}, Train Diversity={train_diversity:.4f}, Train Constraint Loss={constraint_loss_avg:.4f}")
            logger.info(f"Iteration {iteration}: Val Loss={val_loss:.4f}, Val Reward={val_reward:.4f}, Val KL Div={val_kl_div:.4f}, Val Diversity={val_diversity:.4f}")
            if discriminator is not None:
                logger.info(f"Iteration {iteration}: Adversarial Loss={adversarial_loss_avg:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(f"best_model_iter_{iteration}.pth")
                logger.info(f"New best model saved at iteration {iteration}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {iteration} iterations")
                    break
        self.writer.close()
        wandb.finish()

    def validate(self, val_loader: DataLoader, constraints: Optional[dict] = None) -> Tuple[float, float, float, float]:
        self.pretrained_model.eval()
        val_loss = 0
        val_reward = 0
        val_kl_div = 0
        val_diversity = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                x_trajectory = self.sample_trajectory()
                generated_sequences = x_trajectory[-1].argmax(dim=-1)
                reward = self.reward_model(generated_sequences)
                self.recent_rewards.append(reward.mean().item())
                current_alpha = self.adjust_alpha()
                kl_div = self.compute_kl_divergence(x_trajectory)
                diversity_loss = self.compute_diversity_loss(generated_sequences)
                constraint_loss = 0.0
                if constraints:
                    constraint_loss = self.compute_constraint_loss(generated_sequences, constraints)
                loss = -reward.mean() + current_alpha * kl_div + self.diversity_weight * diversity_loss + constraint_loss
                val_loss += loss.item()
                val_reward += reward.mean().item()
                val_kl_div += kl_div.item()
                val_diversity += diversity_loss.item()
        val_loss /= len(val_loader)
        val_reward /= len(val_loader)
        val_kl_div /= len(val_loader)
        val_diversity /= len(val_loader)
        return val_loss, val_reward, val_kl_div, val_diversity

    def save_checkpoint(self, filename: str):
        torch.save({
            'model_state_dict': self.pretrained_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, filename)
        logger.info(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(filename, map_location=self.device)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded from {filename}")

    def generate(self, num_samples: int = 1, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        self.pretrained_model.eval()
        x_trajectory = self.sample_trajectory()
        logits = self.pretrained_model(x_trajectory[-1].argmax(dim=-1), torch.tensor([0.0], device=self.device))
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
        if top_p is not None:
            logits = self.top_p_logits(logits, top_p)
        samples = F.softmax(logits, dim=-1)
        samples = samples.argmax(dim=-1)[:num_samples]
        return samples

    def top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        values, _ = torch.topk(logits, k, dim=-1)
        min_values = values[:, :, -1].unsqueeze(-1).expand_as(logits)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)
        return logits

    def top_p_logits(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('Inf'))
        return logits

def main(args):
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = vars(args)
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    random.seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    train_dataset = DNADataset(data_path=config['train_data'], seq_length=config['seq_length'], augment=True)
    val_dataset = DNADataset(data_path=config['val_data'], seq_length=config['seq_length'], augment=False)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config.get('num_workers', 4))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config.get('num_workers', 4))
    pretrained_model = MaskedDiffusionModel(
        seq_length=config['seq_length'],
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.1)
    ).to(device)
    reward_model = RewardModel(
        seq_length=config['seq_length'],
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size']
    ).to(device)
    if config.get('use_transfer_learning', False):
        try:
            pretrained_weights = torch.load(config['transfer_learning_path'], map_location=device)
            pretrained_model.load_state_dict(pretrained_weights['model_state_dict'], strict=False)
            logger.info("Transfer learning: Pretrained weights loaded successfully.")
        except Exception as e:
            logger.error(f"Transfer learning failed: {e}")
    drakes = DRAKES(
        pretrained_model=pretrained_model,
        reward_model=reward_model,
        alpha=config['alpha'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        num_iterations=config['num_iterations'],
        time_steps=config['time_steps'],
        delta_t=config['delta_t'],
        temperature=config['temperature'],
        device=device,
        scheduler_type=config.get('scheduler_type', 'onecycle'),
        reward_window=config.get('reward_window', 100),
        diversity_weight=config.get('diversity_weight', 0.01)
    )
    if config.get('checkpoint_path'):
        drakes.load_checkpoint(config['checkpoint_path'])
    visualizer = AttentionVisualizer(model=drakes, writer=drakes.writer, device=device)
    discriminator = None
    discriminator_optimizer = None
    discriminator_scheduler = None
    if config.get('use_adversarial_training', False):
        discriminator = Discriminator(vocab_size=config['vocab_size'], hidden_size=config['hidden_size']).to(device)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['discriminator_lr'])
        discriminator_scheduler = OneCycleLR(discriminator_optimizer, max_lr=config['discriminator_lr'], total_steps=config['num_iterations'])
        logger.info("Adversarial training enabled.")
    drakes.train(
        train_loader=train_loader, 
        val_loader=val_loader, 
        discriminator=discriminator, 
        motifs=config.get('motifs'),
        constraints=config.get('constraints')
    )
    final_model_path = os.path.join(config['output_dir'], 'final_model.pth')
    drakes.save_checkpoint(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    generated_sequences = drakes.generate(
        num_samples=config.get('num_generate', 10), 
        top_k=config.get('top_k'), 
        top_p=config.get('top_p')
    )
    logger.info("Generated DNA sequences:")
    for seq in generated_sequences.cpu().numpy():
        print(''.join(['ACGT'[i] for i in seq]))
    if config.get('visualize_attention', False):
        sample_input = torch.randint(0, config['vocab_size'], (1, config['seq_length'])).to(device)
        visualizer.visualize_attention(sample_input, layer=config.get('attention_layer', 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRAKES DNA Sequence Generation with Enhanced Features")
    parser.add_argument('--config', type=str, help='Path to the YAML config file')
    parser.add_argument('--train_data', type=str, default='train_data.npy', help='Path to training data (.npy)')
    parser.add_argument('--val_data', type=str, default='val_data.npy', help='Path to validation data (.npy)')
    parser.add_argument('--seq_length', type=int, default=200, help='Length of DNA sequences')
    parser.add_argument('--vocab_size', type=int, default=4, help='Vocabulary size (A, C, G, T)')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of embeddings')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.001, help='Initial KL divergence weight')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--time_steps', type=int, default=100, help='Number of time steps in diffusion')
    parser.add_argument('--delta_t', type=float, default=0.01, help='Delta t for diffusion')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for Gumbel-Softmax')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save models and logs')
    parser.add_argument('--checkpoint_path', type=str, help='Path to load a checkpoint')
    parser.add_argument('--num_generate', type=int, default=10, help='Number of sequences to generate after training')
    parser.add_argument('--top_k', type=int, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, help='Top-p sampling')
    parser.add_argument('--visualize_attention', action='store_true', help='Visualize attention weights')
    parser.add_argument('--attention_layer', type=int, default=0, help='Transformer layer to visualize attention')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_adversarial_training', action='store_true', help='Use adversarial training with a discriminator')
    parser.add_argument('--discriminator_lr', type=float, default=1e-4, help='Learning rate for the discriminator')
    parser.add_argument('--motifs', type=str, nargs='*', help='List of motifs to attend to')
    parser.add_argument('--use_transfer_learning', action='store_true', help='Use transfer learning from a pretrained model')
    parser.add_argument('--transfer_learning_path', type=str, help='Path to the pretrained model weights')
    parser.add_argument('--constraints', type=str, nargs='*', help='Constraints for sequence generation in key=value format')
    args = parser.parse_args()
    if args.constraints:
        constraints_dict = {}
        for constraint in args.constraints:
            key, value = constraint.split('=')
            try:
                value = float(value)
            except ValueError:
                pass
            constraints_dict[key] = value
        args.constraints = constraints_dict
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
