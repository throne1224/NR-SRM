import asyncio
from protocol import xor_encrypt_decrypt, get_trust_id

HOST = '127.0.0.1'
PORT = 8888

async def main():
    reader, writer = await asyncio.open_connection(HOST, PORT)

    trust_id = get_trust_id()
    encrypted = xor_encrypt_decrypt(trust_id, trust_id)
    writer.write(encrypted.decode().encode())
    await writer.drain()

    response = await reader.read(1024)
    if response == b'OK':
        writer.write(b'Hello Server')
        await writer.drain()
        response = await reader.read(1024)
        print("Received:", response.decode())
    else:
        print("Server rejected connection.")

    writer.close()
    await writer.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())
