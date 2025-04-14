import tensorflow as tf

def custom_flood_loss(y_true, y_pred):
    """
    Custom loss function for flood forecasting using a domain-informed, weighted Huber loss.

    The loss function penalizes prediction errors differently depending on whether the 
    true streamflow value represents a non-flood, flood, or devastated flood condition.

    Parameters:
        y_true (tf.Tensor): Ground truth streamflow values.
        y_pred (tf.Tensor): Predicted streamflow values.

    Returns:
        tf.Tensor: Computed custom loss.
    """

    # Domain-specific flood thresholds (normalized values)
    flood_threshold = 0.0153683       # Flood threshold (FT)
    devastated_threshold = 0.163299   # Devastation threshold (DT)

    # Penalty multipliers for different flood levels
    alpha1 = 2   # Penalty weight for flood level
    alpha2 = 4   # Penalty weight for devastation level
    beta = 0.1   # Regularization term weight
    delta = 0.05 # Delta for Huber loss

    # Compute element-wise Huber loss
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    huber_loss = tf.where(is_small_error, squared_loss, linear_loss)

    # Apply conditional penalties based on streamflow category
    devastated_loss = tf.where(y_true > devastated_threshold, huber_loss, tf.zeros_like(huber_loss))
    flood_loss = tf.where((y_true > flood_threshold) & (y_true <= devastated_threshold), huber_loss, tf.zeros_like(huber_loss))
    non_flood_loss = tf.where(y_true <= flood_threshold, huber_loss, tf.zeros_like(huber_loss))

    # Weighted mean losses for each category
    devastated_loss = tf.reduce_mean(devastated_loss) * alpha2
    flood_loss = tf.reduce_mean(flood_loss) * alpha1
    non_flood_loss = tf.reduce_mean(non_flood_loss)

    # L2 regularization on predictions to ensure smoothness
    regularization_term = beta * tf.reduce_mean(tf.square(y_pred[:, 1:] - y_pred[:, :-1]))

    # Total custom loss
    total_loss = devastated_loss + flood_loss + non_flood_loss + regularization_term

    return total_loss
