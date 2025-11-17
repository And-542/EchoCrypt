import torch

def data_to_binary_vector(data_bytes: bytes, latent_dim: int) -> torch.Tensor:
    """Converts a byte string into a binary latent vector."""
    HEADER_BITS = 8
    max_payload_bytes = (latent_dim - HEADER_BITS) // 8
    if len(data_bytes) > max_payload_bytes:
        raise ValueError(f"Data payload is too long for one chunk. Max length: {max_payload_bytes} bytes.")

    binary_payload = ''.join(format(byte, '08b') for byte in data_bytes)
    length_binary = format(len(data_bytes), f'0{HEADER_BITS}b')
    full_binary_string = length_binary + binary_payload
    binary_values = [1.0 if bit == '1' else -1.0 for bit in full_binary_string]
    
    padding_size = latent_dim - len(binary_values)
    padded_vector = binary_values + [0.0] * padding_size
    return torch.tensor(padded_vector, dtype=torch.float32).unsqueeze(0)

def binary_vector_to_data(vector: torch.Tensor) -> bytes:
    """Converts a binary latent vector back into a byte string."""
    binary_string = ''.join(['1' if val > 0 else '0' for val in vector.squeeze()])
    
    HEADER_BITS = 8
    if len(binary_string) < HEADER_BITS:
        return b""
    
    length_binary = binary_string[:HEADER_BITS]
    try:
        message_length = int(length_binary, 2)
    except ValueError:
        return b"" # Invalid length prefix
    
    total_bits = HEADER_BITS + message_length * 8
    if len(binary_string) < total_bits:
        return b"" # Not enough data

    data_binary = binary_string[HEADER_BITS:total_bits]
    byte_chunks = [data_binary[i:i+8] for i in range(0, len(data_binary), 8)]
    
    try:
        return b"".join([int(b, 2).to_bytes(1, 'big') for b in byte_chunks])
    except (ValueError, OverflowError):
        return b""