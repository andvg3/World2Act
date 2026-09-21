import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionVAE(nn.Module):
    def __init__(self, action_dim=12, latent_dim=32, sequence_len=4, video_channels=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_len = sequence_len
        
        # Optimized for Single Frame (Time=1) input
        self.video_bridger = nn.Sequential(
            nn.Conv3d(video_channels, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(128, latent_dim)
        )
        
        # Encoder (Takes sequence_len=4)
        self.encoder = nn.Sequential(
            nn.Conv1d(action_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # 4 -> 2
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(128 * (sequence_len // 2), latent_dim * 2) 
        )
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * (sequence_len // 2))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(64, action_dim, kernel_size=3, stride=1, padding=1),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _encode_video(self, video_frames):
        """
        Encodes video frames individually.
        Input: (B, C, T, H, W) -> (B, T, C, H, W) -> Stack to (B*T, C, 1, H, W)
        """
        B, C, T, H, W = video_frames.shape
        x = video_frames.permute(0, 2, 1, 3, 4).reshape(-1, C, 1, H, W)
        z = self.video_bridger(x)
        return z.view(B, T, -1)

    def forward(self, action, negative_action_batch, video_latent_x0=None):
        """
        action: (B, 50, 4, 12)
        negative_action_batch: (B, N_neg, 50, 4, 12)
        video_latent_x0: (B, 16, 24, 60, 104) or None
        """
        B, T_chunks, Seq, Dim = action.shape
        _, N_neg, _, _, _ = negative_action_batch.shape
        
        # ==========================================
        # 1. RECONSTRUCTION (Positives AND Negatives)
        # ==========================================
        
        # --- A. Handle Positives ---
        flat_positives = action.view(-1, Seq, Dim) # (B*50, 4, 12)
        
        # Encode Positives
        h_pos = self.encoder(flat_positives.transpose(1, 2))
        mu_pos, logvar_pos = torch.chunk(h_pos, 2, dim=-1) # (B*50, D)
        z_pos = self.reparameterize(mu_pos, logvar_pos)
        
        # Decode Positives
        z_in_pos = self.decoder_input(z_pos).view(-1, 128, Seq // 2)
        recon_pos = self.decoder(z_in_pos).transpose(1, 2)
        
        # --- B. Handle Negatives ---
        # Flatten: (B * N_neg * 50, 4, 12)
        flat_negatives = negative_action_batch.view(-1, Seq, Dim)
        
        # Encode Negatives
        h_neg = self.encoder(flat_negatives.transpose(1, 2))
        mu_neg, logvar_neg = torch.chunk(h_neg, 2, dim=-1) # (B*N_neg*50, D)
        z_neg = self.reparameterize(mu_neg, logvar_neg)
        
        # Decode Negatives
        z_in_neg = self.decoder_input(z_neg).view(-1, 128, Seq // 2)
        recon_neg = self.decoder(z_in_neg).transpose(1, 2)

        # --- C. Compute Reconstruction Loss ---
        loss_pos_recon = F.mse_loss(recon_pos, flat_positives)
        loss_neg_recon = F.mse_loss(recon_neg, flat_negatives)
        
        # You can weight these if necessary, e.g., 1.0 * pos + 0.5 * neg
        recon_loss = loss_pos_recon + loss_neg_recon

        # ==========================================
        # 2. CONTRASTIVE LOSS (On First 24 Chunks)
        # ==========================================
        contrastive_loss = torch.tensor(0.0, device=action.device)
        
        if video_latent_x0 is not None:
            # A. Get Video Embeddings (B, 24, D)
            video_z = self._encode_video(video_latent_x0) 
            video_z = F.normalize(video_z, dim=-1)
            T_vid = video_z.shape[1] # Should be 24
            
            # B. Get Corresponding Positive Embeddings
            # Reshape mu_pos back to (B, 50, D) and slice
            mu_pos_reshaped = mu_pos.view(B, T_chunks, -1)
            anchor_mu = mu_pos_reshaped[:, :T_vid, :] # (B, 24, D)
            anchor_mu = F.normalize(anchor_mu, dim=-1)
            
            # C. Get Corresponding Negative Embeddings
            # Reshape mu_neg back to (B, N_neg, 50, D)
            mu_neg_reshaped = mu_neg.view(B, N_neg, T_chunks, -1)
            
            # Slice first 24 chunks: (B, N_neg, 24, D)
            neg_mu_slice = mu_neg_reshaped[:, :, :T_vid, :]
            
            # Rearrange for Matmul: (B, 24, N_neg, D)
            neg_mu_ready = neg_mu_slice.permute(0, 2, 1, 3)
            neg_mu_ready = F.normalize(neg_mu_ready, dim=-1)

            # D. Compute Contrastive Loss (InfoNCE)
            # Positive Logits: (B, 24, 1)
            pos_logits = (video_z * anchor_mu).sum(dim=-1, keepdim=True)
            
            # Negative Logits: (B, 24, 1, D) @ (B, 24, D, N_neg) -> (B, 24, 1, N_neg)
            neg_logits = torch.matmul(video_z.unsqueeze(2), neg_mu_ready.transpose(2, 3))
            neg_logits = neg_logits.squeeze(2) 
            
            # Combine
            logits = torch.cat([pos_logits, neg_logits], dim=2) # (B, 24, 1+N_neg)
            logits = logits.view(-1, 1 + N_neg) / 0.07
            
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=action.device)
            contrastive_loss = F.cross_entropy(logits, labels)

        # Return both individual stats if you want to log them separately
        return recon_loss, contrastive_loss, mu_pos, logvar_pos