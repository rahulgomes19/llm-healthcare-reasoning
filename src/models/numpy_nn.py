"""
Pure NumPy Neural Network Implementation
Replaces TensorFlow/Keras for cluster environments without GPU support.
"""

import numpy as np
import pickle


class NumpyNeuralNetwork:
    """
    Pure NumPy implementation of a feedforward neural network.
    Architecture: Input -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid)
    """
    
    def __init__(self, input_dim, hidden1=64, hidden2=32, learning_rate=0.001, random_state=42):
        """Initialize neural network with random weights."""
        np.random.seed(random_state)
        
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.learning_rate = learning_rate
        
        # Xavier/Glorot initialization
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden1))
        
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        
        self.W3 = np.random.randn(hidden2, 1) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, 1))
        
        # Adam optimizer parameters
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_W3, self.v_W3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.m_b3, self.v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Adam timestep
    
    def relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative."""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation."""
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Sigmoid derivative."""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X, training=False):
        """
        Forward pass through the network.
        
        Args:
            X: Input data (batch_size, input_dim)
            training: Whether in training mode (for dropout)
        
        Returns:
            predictions: (batch_size, 1) output probabilities
            cache: Dictionary of intermediate values for backprop
        """
        # Layer 1: Dense(64) + ReLU
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Dropout (only during training)
        if training:
            dropout1_mask = np.random.binomial(1, 0.8, size=a1.shape) / 0.8
            a1 = a1 * dropout1_mask
        else:
            dropout1_mask = None
        
        # Layer 2: Dense(32) + ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        
        # Dropout (only during training)
        if training:
            dropout2_mask = np.random.binomial(1, 0.8, size=a2.shape) / 0.8
            a2 = a2 * dropout2_mask
        else:
            dropout2_mask = None
        
        # Layer 3: Dense(1) + Sigmoid
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self.sigmoid(z3)
        
        cache = {
            'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'a3': a3,
            'dropout1_mask': dropout1_mask, 'dropout2_mask': dropout2_mask
        }
        
        return a3, cache
    
    def binary_crossentropy_loss(self, y_true, y_pred):
        """Calculate binary cross-entropy loss."""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, y_true, cache):
        """
        Backward pass (backpropagation).
        
        Args:
            y_true: True labels (batch_size, 1)
            cache: Dictionary from forward pass
        
        Returns:
            gradients: Dictionary of gradients for each parameter
        """
        m = y_true.shape[0]  # batch size
        
        # Output layer gradient
        da3 = cache['a3'] - y_true  # derivative of BCE with sigmoid
        
        # Layer 3 gradients
        dW3 = np.dot(cache['a2'].T, da3) / m
        db3 = np.sum(da3, axis=0, keepdims=True) / m
        
        # Layer 2 gradients
        da2 = np.dot(da3, self.W3.T)
        if cache['dropout2_mask'] is not None:
            da2 = da2 * cache['dropout2_mask']
        dz2 = da2 * self.relu_derivative(cache['z2'])
        dW2 = np.dot(cache['a1'].T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Layer 1 gradients
        da1 = np.dot(dz2, self.W2.T)
        if cache['dropout1_mask'] is not None:
            da1 = da1 * cache['dropout1_mask']
        dz1 = da1 * self.relu_derivative(cache['z1'])
        dW1 = np.dot(cache['X'].T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    
    def adam_update(self, gradients):
        """Update weights using Adam optimizer."""
        self.t += 1
        
        # Update W1, b1
        self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * gradients['dW1']
        self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (gradients['dW1'] ** 2)
        m_W1_hat = self.m_W1 / (1 - self.beta1 ** self.t)
        v_W1_hat = self.v_W1 / (1 - self.beta2 ** self.t)
        self.W1 -= self.learning_rate * m_W1_hat / (np.sqrt(v_W1_hat) + self.epsilon)
        
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * gradients['db1']
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (gradients['db1'] ** 2)
        m_b1_hat = self.m_b1 / (1 - self.beta1 ** self.t)
        v_b1_hat = self.v_b1 / (1 - self.beta2 ** self.t)
        self.b1 -= self.learning_rate * m_b1_hat / (np.sqrt(v_b1_hat) + self.epsilon)
        
        # Update W2, b2
        self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * gradients['dW2']
        self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (gradients['dW2'] ** 2)
        m_W2_hat = self.m_W2 / (1 - self.beta1 ** self.t)
        v_W2_hat = self.v_W2 / (1 - self.beta2 ** self.t)
        self.W2 -= self.learning_rate * m_W2_hat / (np.sqrt(v_W2_hat) + self.epsilon)
        
        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * gradients['db2']
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (gradients['db2'] ** 2)
        m_b2_hat = self.m_b2 / (1 - self.beta1 ** self.t)
        v_b2_hat = self.v_b2 / (1 - self.beta2 ** self.t)
        self.b2 -= self.learning_rate * m_b2_hat / (np.sqrt(v_b2_hat) + self.epsilon)
        
        # Update W3, b3
        self.m_W3 = self.beta1 * self.m_W3 + (1 - self.beta1) * gradients['dW3']
        self.v_W3 = self.beta2 * self.v_W3 + (1 - self.beta2) * (gradients['dW3'] ** 2)
        m_W3_hat = self.m_W3 / (1 - self.beta1 ** self.t)
        v_W3_hat = self.v_W3 / (1 - self.beta2 ** self.t)
        self.W3 -= self.learning_rate * m_W3_hat / (np.sqrt(v_W3_hat) + self.epsilon)
        
        self.m_b3 = self.beta1 * self.m_b3 + (1 - self.beta1) * gradients['db3']
        self.v_b3 = self.beta2 * self.v_b3 + (1 - self.beta2) * (gradients['db3'] ** 2)
        m_b3_hat = self.m_b3 / (1 - self.beta1 ** self.t)
        v_b3_hat = self.v_b3 / (1 - self.beta2 ** self.t)
        self.b3 -= self.learning_rate * m_b3_hat / (np.sqrt(v_b3_hat) + self.epsilon)
    
    def train_batch(self, X_batch, y_batch):
        """Train on a single batch."""
        # Forward pass
        y_pred, cache = self.forward(X_batch, training=True)
        
        # Calculate loss
        loss = self.binary_crossentropy_loss(y_batch, y_pred)
        
        # Backward pass
        gradients = self.backward(y_batch, cache)
        
        # Update weights with Adam
        self.adam_update(gradients)
        
        return loss
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=200, batch_size=256, 
            patience=20, verbose=True):
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
            verbose: Whether to print progress
        
        Returns:
            history: Dictionary with training history
        """
        y_train = y_train.reshape(-1, 1)
        if y_val is not None:
            y_val = y_val.reshape(-1, 1)
        
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Train on batches
            epoch_losses = []
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                loss = self.train_batch(X_batch, y_batch)
                epoch_losses.append(loss)
            
            # Calculate training metrics
            train_loss = np.mean(epoch_losses)
            train_pred, _ = self.forward(X_train, training=False)
            train_acc = np.mean((train_pred > 0.5).astype(int) == y_train)
            
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred, _ = self.forward(X_val, training=False)
                val_loss = self.binary_crossentropy_loss(y_val, val_pred)
                val_acc = np.mean((val_pred > 0.5).astype(int) == y_val)
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    best_weights = {
                        'W1': self.W1.copy(), 'b1': self.b1.copy(),
                        'W2': self.W2.copy(), 'b2': self.b2.copy(),
                        'W3': self.W3.copy(), 'b3': self.b3.copy()
                    }
                else:
                    patience_counter += 1
                    
                    # Reduce learning rate on plateau (every 8 epochs without improvement)
                    if patience_counter % 8 == 0 and patience_counter > 0:
                        self.learning_rate *= 0.3
                        self.learning_rate = max(self.learning_rate, 1e-5)
                        if verbose:
                            print(f"  -> Reducing learning rate to {self.learning_rate:.6f}")
                    
                    if patience_counter >= patience:
                        if verbose:
                            print(f"\nEarly stopping at epoch {epoch+1}")
                        # Restore best weights
                        if best_weights is not None:
                            self.W1 = best_weights['W1']
                            self.b1 = best_weights['b1']
                            self.W2 = best_weights['W2']
                            self.b2 = best_weights['b2']
                            self.W3 = best_weights['W3']
                            self.b3 = best_weights['b3']
                        break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features (n_samples, input_dim)
        
        Returns:
            predictions: (n_samples, 1) probabilities
        """
        predictions, _ = self.forward(X, training=False)
        return predictions
    
    def save(self, filepath):
        """Save model weights to file."""
        weights = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'input_dim': self.input_dim,
            'hidden1': self.hidden1,
            'hidden2': self.hidden2,
            'learning_rate': self.learning_rate
        }
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model weights from file."""
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        
        # Create model with saved architecture
        model = cls(
            input_dim=weights['input_dim'],
            hidden1=weights['hidden1'],
            hidden2=weights['hidden2'],
            learning_rate=weights['learning_rate']
        )
        
        # Load weights
        model.W1 = weights['W1']
        model.b1 = weights['b1']
        model.W2 = weights['W2']
        model.b2 = weights['b2']
        model.W3 = weights['W3']
        model.b3 = weights['b3']
        
        return model
