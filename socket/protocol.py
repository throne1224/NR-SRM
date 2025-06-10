import os

def xor_encrypt_decrypt(data: str, key: str) -> bytes:
    return bytes([ord(c) ^ ord(key[i % len(key)]) for i, c in enumerate(data)])

def get_trust_id() -> str:
    trust_id = os.getenv("TRUST_ID")
    if not trust_id:
        raise EnvironmentError("TRUST_ID 环境变量未设置")
    return trust_id
