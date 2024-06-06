import tensorflow as tf

# Check for available devices (including GPU)
print(tf.config.list_physical_devices())

# Enable memory growth for the first GPU (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
logical_gpus = tf.config.list_logical_devices('GPU')

if logical_gpus:
  print("GPU is available!")
  device = tf.device('/device:GPU:0')  # Set device to first GPU
else:
  print("GPU is not available. Training on CPU.")
  device = tf.device('/device:CPU:0')  # Fallback to CPU