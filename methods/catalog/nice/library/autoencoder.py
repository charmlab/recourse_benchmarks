"""
Simple Autoencoder for plausibility measurement in NICE(plaus)

Based on standard architecture used in counterfactual literature
(CFproto, similar methods) and the benchmark's existing test patterns.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from math import ceil


class AutoEncoder:
    """
    Autoencoder to measure plausibility via reconstruction error.
    
    Trains a simple feedforward autoencoder on training data.
    Lower reconstruction error = more plausible (closer to data manifold).
    
    Parameters
    ----------
    X_train : np.ndarray
        Training data (N x M)
    cat_feat_idx : list
        Indices of categorical features (not used in current implementation)
    num_feat_idx : list
        Indices of numerical features (not used in current implementation)
    encoding_dim : int, default=32
        Dimension of encoded (latent) representation
    epochs : int, default=50
        Number of training epochs
    batch_size : int, default=32
        Batch size for training
    verbose : int, default=0
        Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
    
    Attributes
    ----------
    autoencoder : keras.Model
        Trained autoencoder model
        
    Notes
    -----
    Architecture: input → 64 (ReLU) → encoding_dim (ReLU) → 64 (ReLU) → output (Sigmoid)
    Loss: Mean Squared Error (MSE)
    Optimizer: Adam
    
    This is a standard architecture used in counterfactual literature for
    measuring plausibility through reconstruction error.
    """
    
    def __init__(
        self, 
        X_train: np.ndarray,
        cat_feat_idx: list = None,
        num_feat_idx: list = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 5,
        verbose: int = 0
    ):
        self.cat_feat_idx = cat_feat_idx if cat_feat_idx is not None else []
        self.num_feat_idx = num_feat_idx if num_feat_idx is not None else []
        
        self.input_dim = X_train.shape[1]  # ✅ FIXED! Number of features, not instances
        self.batch_size = batch_size
        
        # Sizes
        latent_size = 2 if int(ceil(self.input_dim / 4)) < 4 else 4
        hidden_1 = int(ceil(self.input_dim / 2))
        hidden_2 = int(ceil(self.input_dim / 4))

        # Build autoencoder architecture
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder: input → input/2 → input/4 → latent
        encoded = layers.Dense(hidden_1, activation='tanh', name='encoder_1')(input_layer)
        encoded = layers.Dense(hidden_2, activation='tanh', name='encoder_2')(encoded)
        encoded = layers.Dense(latent_size, activation='tanh', name='latent')(encoded)
        
        # Decoder: latent → input/4 → input/2 → input (symmetric)
        decoded = layers.Dense(hidden_2, activation='tanh', name='decoder_1')(encoded)
        decoded = layers.Dense(hidden_1, activation='tanh', name='decoder_2')(decoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid', name='output')(decoded)
        
        # Create model
        self.autoencoder = keras.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Setup callbacks
        callbacks = []
        if early_stopping:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=verbose,
                mode='min'
            )
            callbacks.append(early_stop)

        # Train autoencoder on training data
        self.autoencoder.fit(
            X_train, 
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
            validation_split=validation_split,
            callbacks=callbacks
        )
        
        if verbose > 0:
            final_epoch = len(self.history.history['loss'])
            final_loss = self.history.history['loss'][-1]
            final_val_loss = self.history.history['val_loss'][-1]
            
            print(f"\n{'='*70}")
            print("Autoencoder Training Summary")
            print(f"{'='*70}")
            print(f"Training instances:    {X_train.shape[0]}")
            print(f"Input dimension:       {self.input_dim}")
            print(f"Architecture:          {self.input_dim} → {hidden_1} → {hidden_2} → "
                  f"{latent_size} → {hidden_2} → {hidden_1} → {self.input_dim}")
            print(f"Epochs completed:      {final_epoch}/{epochs}")
            print(f"Final training loss:   {final_loss:.6f}")
            print(f"Final validation loss: {final_val_loss:.6f}")
            
            if early_stopping and final_epoch < epochs:
                print(f"Early stopping triggered (patience={patience})")
            
            print(f"{'='*70}\n")
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for instances.
        
        This is the main interface used by PlausibilityReward.
        
        Parameters
        ----------
        X : np.ndarray
            Instances to evaluate (N x M)
        
        Returns
        -------
        np.ndarray
            Reconstruction errors, one per instance (N,)
            Higher error = less plausible (farther from data manifold)
        """
        # Get reconstructions
        X_reconstructed = self.autoencoder.predict(X, verbose=0)
        
        # Calculate Mean Squared Error per instance
        errors = np.mean(np.square(X - X_reconstructed), axis=1)
        
        return errors
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct instances through the autoencoder.
        """
        return self.autoencoder.predict(X, verbose=0)
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Get the latent (encoded) representation of instances."""
        encoder = keras.Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.get_layer('latent').output
        )
        return encoder.predict(X, verbose=0)